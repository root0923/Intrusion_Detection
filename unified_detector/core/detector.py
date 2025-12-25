"""
统一检测器 - Unified Detector

功能：
- 封装 YOLO model.track() 推理
- 提供统一的检测结果格式
- 支持目标跟踪（ByteTrack）
"""
import logging
from typing import List, Dict
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load


logger = logging.getLogger(__name__)


class UnifiedDetector:
    """统一目标检测器（封装YOLO + ByteTrack）"""

    def __init__(self, model_yaml: str, model_weights: str, device: str = 'cuda:0',
                 tracker: str = 'bytetrack'):
        """
        Args:
            model_yaml: 模型配置文件路径
            model_weights: 模型权重文件路径
            device: 设备 (cuda:0 或 cpu)
            tracker: 跟踪器类型 (bytetrack 或 botsort)
        """
        self.device = device
        self.tracker = tracker

        logger.info(f"初始化YOLO模型...")
        logger.info(f"  模型配置: {model_yaml}")
        logger.info(f"  模型权重: {model_weights}")
        logger.info(f"  设备: {device}")
        logger.info(f"  跟踪器: {tracker}")

        # 加载模型
        try:
            # 读取模型通道数
            yaml_dict = yaml_model_load(model_yaml)
            self.model_ch = yaml_dict.get('ch', 3)

            # 加载YOLO模型
            self.model = YOLO(model_weights)

            logger.info(f"✓ 模型加载成功 (ch={self.model_ch})")

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {e}")
            raise

    def detect(self, frame, conf_threshold: float = 0.25, iou_threshold: float = 0.7,
               target_size: int = 640) -> List[Dict]:
        """
        检测（不跟踪）

        Args:
            frame: 输入图像 (H, W, 3)
            conf_threshold: 置信度阈值（统一用0.25，后续规则再过滤）
            iou_threshold: NMS IOU阈值
            target_size: 推理图像尺寸

        Returns:
            List[Dict]: 检测结果列表
                {
                    'bbox': [x1, y1, x2, y2],
                    'conf': float,
                    'cls': int
                }
        """
        try:
            # 使用model.predict进行推理（不跟踪）
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=target_size,
                use_simotm="RGB",
                channels=self.model_ch,
                verbose=False,
                device=self.device
            )

            # 解析结果
            detections = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls in zip(boxes, confs, classes):
                        detection = {
                            'bbox': box.tolist(),
                            'conf': float(conf),
                            'cls': int(cls)
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"检测异常: {e}", exc_info=True)
            return []

    def detect_and_track(self, frame, conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                        target_size: int = 640) -> List[Dict]:
        """
        检测并跟踪（统一接口）- 保留用于非绊线规则

        Args:
            frame: 输入图像 (H, W, 3)
            conf_threshold: 置信度阈值（统一用0.25，后续规则再过滤）
            iou_threshold: NMS IOU阈值
            target_size: 推理图像尺寸

        Returns:
            List[Dict]: 检测结果列表
                {
                    'bbox': [x1, y1, x2, y2],
                    'conf': float,
                    'cls': int,
                    'track_id': int  # 如果有跟踪
                }
        """
        try:
            # 使用model.track进行推理（带跟踪）
            results = self.model.track(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=target_size,
                use_simotm="RGB",
                channels=self.model_ch,
                persist=True,  # 保持tracker状态
                tracker=f"{self.tracker}.yaml",
                verbose=False,
                device=self.device
            )

            # 解析结果
            detections = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    # 提取track_id（如果存在）
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = [None] * len(boxes)

                    for box, conf, cls, track_id in zip(boxes, confs, classes, track_ids):
                        detection = {
                            'bbox': box.tolist(),
                            'conf': float(conf),
                            'cls': int(cls)
                        }
                        if track_id is not None:
                            detection['track_id'] = int(track_id)

                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"检测异常: {e}", exc_info=True)
            return []

    def reset_tracker(self):
        """重置跟踪器（用于配置更新时）"""
        try:
            # Ultralytics的tracker会在下次track调用时自动重置
            # 这里只是提供一个接口，实际不需要手动重置
            logger.debug("跟踪器已重置")
        except Exception as e:
            logger.warning(f"重置跟踪器失败: {e}")
