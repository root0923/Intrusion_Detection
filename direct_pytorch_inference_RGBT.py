import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import yaml

from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import ops


class RGBTDetector:

    def __init__(self, yaml_path, weights_path, device='cuda:0'):
        """

        Args:
            yaml_path: 模型YAML配置文件路径
            weights_path: 权重文件(.pt)路径
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.weights_path = weights_path

        # 1. 从YAML创建模型结构
        print(f"\n从YAML创建模型: {yaml_path}")
        self.model = self._build_model_from_yaml(yaml_path)

        # 2. 加载权重
        print(f"加载权重: {weights_path}")
        self._load_weights(weights_path)

        # 3. 设置为评估模式
        self.model.eval()
        self.model.to(self.device)

        # 4. 保存类别数
        if hasattr(self.model, 'yaml'):
            self.nc = self.model.yaml.get('nc', 80)
        elif hasattr(self.model, 'nc'):
            self.nc = self.model.nc
        else:
            # 从checkpoint读取
            ckpt = torch.load(weights_path, map_location='cpu')
            if 'model' in ckpt and hasattr(ckpt['model'], 'yaml'):
                self.nc = ckpt['model'].yaml.get('nc', 80)
            else:
                self.nc = 80  # 默认COCO类别数

        print(f"  类别数: {self.nc}")
        print("✓ 模型初始化完成\n")

    def _build_model_from_yaml(self, yaml_path):
        """从YAML文件构建模型"""
        yaml_dict = yaml_model_load(yaml_path)

        # 从权重文件中获取正确的nc（类别数）
        ckpt = torch.load(self.weights_path, map_location='cpu')
        if 'model' in ckpt and hasattr(ckpt['model'], 'yaml'):
            nc = ckpt['model'].yaml.get('nc', 80)
            ch = ckpt['model'].yaml.get('ch', 3)
            print(f"  从权重文件读取: nc={nc}, ch={ch}")
        else:
            nc = yaml_dict.get('nc', 80)
            ch = yaml_dict.get('ch', 3)

        # 创建DetectionModel
        model = DetectionModel(cfg=yaml_dict, ch=ch, nc=nc, verbose=False)

        return model

    def _load_weights(self, weights_path):
        """加载权重到模型"""
        ckpt = torch.load(weights_path, map_location='cpu')

        if 'model' in ckpt:
            state_dict = ckpt['model'].state_dict()
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict, strict=True)

    def preprocess(self, rgb_img, infrared_img, target_size=640, stride=32, auto=True):
        """
        预处理RGB和红外图像（与Ultralytics LetterBox完全一致）

        Args:
            rgb_img: RGB图像 (H, W, 3) BGR格式
            infrared_img: 红外图像 (H, W, 3) 或 (H, W)
            target_size: 目标尺寸
            stride: 模型步长，默认32
            auto: 是否使用最小矩形模式（与Ultralytics predict一致）

        Returns:
            tensor: (1, 4, H, W) RGBT tensor
            scale: 缩放比例
            pad: padding信息
        """
        # 保存原始尺寸
        h0, w0 = rgb_img.shape[:2]

        # 转换红外为单通道
        if len(infrared_img.shape) == 3:
            infrared_gray = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)
        else:
            infrared_gray = infrared_img

        # 确保尺寸一致
        if rgb_img.shape[:2] != infrared_gray.shape[:2]:
            infrared_gray = cv2.resize(infrared_gray, (w0, h0))

        # 计算缩放比例（与Ultralytics LetterBox一致）
        r = min(target_size / h0, target_size / w0)

        # 计算新尺寸（使用round而不是int截断）
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))

        # 缩放图像
        if r != 1:
            # 根据是否放大选择插值方式
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=interp)
            infrared_gray = cv2.resize(infrared_gray, (new_w, new_h), interpolation=interp)

        # 计算padding
        dw = target_size - new_w
        dh = target_size - new_h

        # auto模式：padding只取stride的余数（与Ultralytics predict模式一致）
        if auto:
            dw = np.mod(dw, stride)
            dh = np.mod(dh, stride)

        # 居中padding（与Ultralytics LetterBox center=True一致）
        dw /= 2
        dh /= 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        rgb_img = cv2.copyMakeBorder(rgb_img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        infrared_gray = cv2.copyMakeBorder(infrared_gray, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=114)

        # 合并为RGBT (4通道) - BGR先合并，后续转RGB
        b, g, r_ch = cv2.split(rgb_img)
        rgbt = np.stack([b, g, r_ch, infrared_gray], axis=2)  # (H, W, 4) BGR + Infrared

        # 转换为tensor并做BGR->RGB转换（与Ultralytics predictor.preprocess一致）
        rgbt = rgbt.transpose(2, 0, 1)  # (4, H, W)
        # BGR to RGB: 前3通道反转
        rgbt[:3] = rgbt[:3][::-1]  # BGR -> RGB
        rgbt = np.ascontiguousarray(rgbt)
        rgbt = torch.from_numpy(rgbt).float()
        rgbt = rgbt.unsqueeze(0)  # (1, 4, H, W)
        rgbt = rgbt / 255.0  # 归一化到0-1

        return rgbt, r, (top, left)

    @torch.no_grad()
    def detect(self, rgb_img, infrared_img, conf_thresh=0.25, iou_thresh=0.7,
               target_size=640, auto=True):
        """
        检测函数

        Args:
            rgb_img: RGB图像 (H, W, 3)
            infrared_img: 红外图像 (H, W, 3)或(H, W)
            conf_thresh: 置信度阈值
            iou_thresh: NMS的IOU阈值（默认0.7与Ultralytics一致）
            target_size: 推理尺寸
            auto: 是否使用auto模式（与Ultralytics predict一致）

        Returns:
            detections: list of dict, 每个dict包含 {'box', 'conf', 'cls'}
        """
        # 保存原始尺寸
        orig_h, orig_w = rgb_img.shape[:2]

        # 预处理
        input_tensor, scale, (pad_top, pad_left) = self.preprocess(
            rgb_img, infrared_img, target_size, auto=auto)

        input_tensor = input_tensor.to(self.device)

        # 前向推理
        predictions = self.model(input_tensor)  # (1, num_anchors, 4+nc)

        # NMS后处理
        detections = self.postprocess(
            predictions,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            orig_shape=(orig_h, orig_w),
            scale=scale,
            pad=(pad_top, pad_left)
        )

        return detections

    def postprocess(self, predictions, conf_thresh, iou_thresh,
                   orig_shape, scale, pad):
        """
        后处理：NMS + 坐标还原

        Args:
            predictions: 模型输出 (1, num_anchors, 4+nc)
            conf_thresh: 置信度阈值
            iou_thresh: IOU阈值
            orig_shape: 原始图像尺寸 (H, W)
            scale: 缩放比例
            pad: padding (top, left)

        Returns:
            detections: list of dict
        """
        # 使用Ultralytics的NMS函数
        predictions = ops.non_max_suppression(
            predictions,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            nc=self.nc  # 使用保存的类别数
        )

        detections = []
        for pred in predictions:  # 遍历batch（这里只有1张图）
            if pred is None or len(pred) == 0:
                continue

            # pred: (N, 6) [x1, y1, x2, y2, conf, cls]
            pred = pred.cpu().numpy()

            # 坐标还原（去掉padding和缩放）
            pad_top, pad_left = pad
            for det in pred:
                x1, y1, x2, y2, conf, cls = det

                # 去除padding
                x1 = (x1 - pad_left) / scale
                y1 = (y1 - pad_top) / scale
                x2 = (x2 - pad_left) / scale
                y2 = (y2 - pad_top) / scale

                # 裁剪到原始图像范围
                x1 = max(0, min(x1, orig_shape[1]))
                y1 = max(0, min(y1, orig_shape[0]))
                x2 = max(0, min(x2, orig_shape[1]))
                y2 = max(0, min(y2, orig_shape[0]))

                detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': float(conf),
                    'cls': int(cls)
                })

        return detections

    def visualize(self, image, detections, class_names=None):
        """
        可视化检测结果

        Args:
            image: 原始图像
            detections: 检测结果
            class_names: 类别名称字典

        Returns:
            vis_image: 可视化后的图像
        """
        vis_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['conf']
            cls = det['cls']

            # 绘制框
            color = self._get_color(cls)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 标签
            cls_name = class_names.get(cls, str(cls)) if class_names else str(cls)
            label = f'{cls_name} {conf:.2f}'

            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10),
                         (x1 + label_w, y1), color, -1)

            # 标签文字
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_image

    def _get_color(self, class_id):
        """根据类别生成颜色"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))


def test_example():
    """测试示例"""
    print("=" * 60)
    print("底层RGBT检测器测试")
    print("=" * 60)

    # ========== 配置 ==========
    YAML_PATH = "ultralytics/cfg/models/11-RGBT/yolo11l-RGBT-midfusion-P3.yaml"
    WEIGHTS_PATH = "data/LLVIP-yolo11l-RGBT-midfusion-P3-e300-16-pretrained-.pt"
    DEVICE = 'cuda:0'

    CONF_THRESH = 0.25
    IOU_THRESH = 0.7
    TARGET_SIZE = 640

    # 类别名称（根据你的数据集修改）
    CLASS_NAMES = {0: 'person'}

    # ========== 初始化检测器 ==========
    detector = RGBTDetector(YAML_PATH, WEIGHTS_PATH, DEVICE)

    # ========== 测试单张图片 ==========
    print("\n测试单张图片...")
    # rgb_img = cv2.imread('data/dataset/visible/j118.jpg')
    # infrared_img = cv2.imread('data/dataset/infrared/j118.jpg')
    rgb_img = cv2.imread('roi_masked_v_n286.jpg')
    infrared_img = cv2.imread('roi_masked_i_n286.jpg')

    # 检测
    detections = detector.detect(
        rgb_img, infrared_img,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH,
        target_size=TARGET_SIZE
    )

    print(f"检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections):
        print(f"  {i+1}. Box: {det['box']}, Conf: {det['conf']:.3f}, "
              f"Class: {CLASS_NAMES.get(det['cls'], det['cls'])}")

    # 可视化
    vis_image = detector.visualize(rgb_img, detections, CLASS_NAMES)
    cv2.imwrite('detection_result.jpg', vis_image)
    print("\n✓ 结果已保存: detection_result.jpg")

    # # ========== 测试RTSP流（实时） ==========
    # print("\n" + "=" * 60)
    # print("测试RTSP实时流...")
    # print("=" * 60)

    # # RTSP地址
    # VISIBLE_RTSP = "rtsp://admin:admin123@192.168.1.100:554/channel1"
    # INFRARED_RTSP = "rtsp://admin:admin123@192.168.1.100:554/channel2"

    # # 打开视频流
    # cap_vis = cv2.VideoCapture(VISIBLE_RTSP)
    # cap_inf = cv2.VideoCapture(INFRARED_RTSP)

    # if not (cap_vis.isOpened() and cap_inf.isOpened()):
    #     print("✗ 无法连接到摄像头")
    #     return

    # print("✓ 摄像头连接成功，按 'q' 退出\n")

    # frame_count = 0
    # while True:
    #     # 读取帧
    #     ret_vis, frame_vis = cap_vis.read()
    #     ret_inf, frame_inf = cap_inf.read()

    #     if not (ret_vis and ret_inf):
    #         break

    #     # 检测
    #     detections = detector.detect(
    #         frame_vis, frame_inf,
    #         conf_thresh=CONF_THRESH,
    #         iou_thresh=IOU_THRESH,
    #         target_size=TARGET_SIZE
    #     )

    #     # 可视化
    #     vis_rgb = detector.visualize(frame_vis, detections, CLASS_NAMES)
    #     vis_inf = detector.visualize(frame_inf, detections, CLASS_NAMES)

    #     # 显示
    #     display = np.hstack([vis_rgb, vis_inf])
    #     cv2.imshow('RGBT Detection', display)

    #     frame_count += 1
    #     if frame_count % 30 == 0:
    #         print(f"Frame {frame_count}, 检测数: {len(detections)}")

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap_vis.release()
    # cap_inf.release()
    # cv2.destroyAllWindows()

    # print(f"\n✓ 处理完成，总帧数: {frame_count}")


if __name__ == '__main__':
    test_example()
