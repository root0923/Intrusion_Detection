"""
ROI坐标标定工具
从视频中提取首帧，交互式绘制多个多边形ROI区域，保存配置文件
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple


class VideoROISelector:
    """视频ROI选择器"""

    def __init__(self, frame: np.ndarray):
        """
        Args:
            frame: 视频帧
        """
        self.frame = frame.copy()
        self.image = frame.copy()
        self.clone = self.image.copy()
        self.all_rois = []  # 存储多个ROI: [[(x1,y1), (x2,y2), ...], ...]
        self.current_roi = []  # 当前正在绘制的ROI点
        self.drawing = False
        self.roi_colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
        ]

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调：绘制多个多边形ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_roi.append((x, y))
            cv2.circle(self.clone, (x, y), 5, (0, 255, 0), -1)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 实时显示绘制轨迹
            temp_image = self.image.copy()
            self._draw_existing_rois(temp_image)

            if len(self.current_roi) > 0:
                # 绘制当前ROI
                for i, pt in enumerate(self.current_roi):
                    cv2.circle(temp_image, pt, 5, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(temp_image, self.current_roi[i-1], pt, (0, 255, 0), 2)

                # 预览闭合
                if len(self.current_roi) > 1:
                    cv2.line(temp_image, self.current_roi[-1], (x, y), (0, 255, 0), 2)

            self.clone = temp_image

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键完成当前ROI
            if len(self.current_roi) >= 3:
                self.all_rois.append(self.current_roi.copy())
                self.current_roi = []
                self.drawing = False
                self._refresh_display()
                print(f"✓ 已添加ROI-{len(self.all_rois)}，包含 {len(self.all_rois[-1])} 个点")

    def _draw_existing_rois(self, image):
        """绘制所有已存在的ROI"""
        for i, roi_points in enumerate(self.all_rois):
            color = self.roi_colors[i % len(self.roi_colors)]

            # 绘制填充区域（半透明）
            overlay = image.copy()
            cv2.fillPoly(overlay, [np.array(roi_points)], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

            # 绘制边界
            cv2.polylines(image, [np.array(roi_points)], True, color, 3)

            # 标注ROI编号
            if roi_points:
                center_x = sum(p[0] for p in roi_points) // len(roi_points)
                center_y = sum(p[1] for p in roi_points) // len(roi_points)
                cv2.putText(image, f"ROI-{i+1}", (center_x-30, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _refresh_display(self):
        """刷新显示"""
        self.clone = self.image.copy()
        self._draw_existing_rois(self.clone)

    def select_rois(self) -> List[List[Tuple[int, int]]]:
        """
        交互式选择多个ROI

        Returns:
            List[List[Tuple]]: ROI列表，每个ROI是顶点坐标列表
        """
        window_name = "ROI区域标定 (ESC退出)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("ROI区域标定系统")
        print("=" * 60)
        print("操作说明:")
        print("  1. 左键点击：添加多边形顶点")
        print("  2. 右键点击：完成当前ROI（至少3个顶点）")
        print("  3. 按 'd'：删除上一个ROI")
        print("  4. 按 's'：保存所有ROI并退出")
        print("  5. 按 'r'：重置所有ROI")
        print("  6. 按 'q'：退出（不保存）")
        print("=" * 60 + "\n")

        while True:
            cv2.imshow(window_name, self.clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and self.all_rois:
                # 保存并退出
                break
            elif key == ord('r'):
                # 重置
                self.all_rois = []
                self.current_roi = []
                self.clone = self.image.copy()
                print("✓ 已重置所有ROI")
            elif key == ord('d'):
                # 删除上一个ROI
                if self.all_rois:
                    removed = self.all_rois.pop()
                    self._refresh_display()
                    print(f"✓ 已删除ROI-{len(self.all_rois)+1}")
                else:
                    print("✗ 没有可删除的ROI")
            elif key == ord('q') or key == 27:  # q或ESC
                # 退出
                print("\n✗ 用户取消")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return self.all_rois


def extract_first_frame(video_path: str) -> Tuple[np.ndarray, int, int]:
    """
    从视频中提取第一帧

    Args:
        video_path: 视频路径或摄像头ID

    Returns:
        (frame, width, height): 首帧图像和尺寸
    """
    # 尝试作为文件路径打开
    cap = cv2.VideoCapture(video_path)

    # 如果失败，尝试作为摄像头ID
    if not cap.isOpened():
        try:
            video_id = int(video_path)
            cap = cv2.VideoCapture(video_id)
        except ValueError:
            pass

    if not cap.isOpened():
        raise ValueError(f"无法打开视频源: {video_path}")

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"无法从视频中读取帧: {video_path}")

    height, width = frame.shape[:2]
    cap.release()

    print(f"✓ 视频信息: {width}x{height}")
    return frame, width, height


def save_roi_config(rois: List[List[Tuple[int, int]]],
                    image_width: int,
                    image_height: int,
                    output_path: str):
    """
    保存ROI配置到JSON文件

    Args:
        rois: ROI列表
        image_width: 图像宽度
        image_height: 图像高度
        output_path: 输出文件路径
    """
    config = {
        "image_width": image_width,
        "image_height": image_height,
        "rois": rois
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✓ ROI配置已保存到: {output_path}")
    return output_path


def print_summary(rois: List, output_path: str, video_source: str):
    """打印配置摘要和下一步说明"""
    print("\n" + "=" * 60)
    print("配置摘要")
    print("=" * 60)
    print(f"ROI数量: {len(rois)}")
    for i, roi in enumerate(rois):
        print(f"  ROI-{i+1}: {len(roi)} 个顶点")
    print(f"配置文件: {output_path}")
    print("=" * 60)

    print("\n下一步：运行区域入侵检测")
    print("=" * 60)
    print(f"python regional_intrusion/intrusion_detector.py \\")
    print(f"    --source {video_source} \\")
    print(f"    --config {output_path} \\")
    print(f"    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \\")
    print(f"    --show --save")
    print("=" * 60 + "\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ROI坐标标定工具 - 从视频中提取首帧并标定ROI区域'
    )

    parser.add_argument('--source', type=str, required=True,
                       help='视频路径、摄像头ID(0,1,...)或RTSP地址')
    parser.add_argument('--output', type=str, default='area_intrusion/roi_config.json',
                       help='输出配置文件路径 (默认: regional_intrusion/roi_config.json)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("ROI坐标标定工具")
    print("=" * 60)

    # 1. 提取视频首帧
    print(f"\n[1/3] 提取视频首帧: {args.source}")
    try:
        frame, width, height = extract_first_frame(args.source)
    except Exception as e:
        print(f"✗ 错误: {e}")
        return

    # 2. 标定ROI区域
    print(f"\n[2/3] 标定ROI区域...")
    selector = VideoROISelector(frame)
    rois = selector.select_rois()

    if rois is None or len(rois) == 0:
        print("✗ 未选择任何ROI区域")
        return

    # 3. 保存配置
    print(f"\n[3/3] 保存配置...")
    save_roi_config(rois, width, height, args.output)

    # 4. 打印摘要
    print_summary(rois, args.output, args.source)


if __name__ == '__main__':
    main()
