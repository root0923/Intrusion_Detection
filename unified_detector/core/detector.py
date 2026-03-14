"""
统一检测器 - Unified Detector

功能：
- 封装 YOLO model.track() 推理
- 提供统一的检测结果格式
- 支持目标跟踪（ByteTrack）
- 支持 RGB 和 IR 模型（.pt 或 .engine）

使用示例：
    # RGB 模型（可见光）
    detector = UnifiedDetector(
        model_yaml="ultralytics/cfg/models/11/yolo11m.yaml",
        model_weights="runs/finetune_V_3classes/weights/last.engine",
        device="cuda:0",
        tracker="bytetrack",
        channels=3,
        use_simotm='RGB'
    )

    # IR 模型（红外）
    detector = UnifiedDetector(
        model_yaml="ultralytics/cfg/models/11/yolo11m.yaml",
        model_weights="runs/finetuneNegIR2/weights/last.engine",
        device="cuda:0",
        tracker="bytetrack",
        channels=3,
        use_simotm='SimOTMBBS'
    )
"""
import logging
from typing import List, Dict
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load


logger = logging.getLogger(__name__)


class UnifiedDetector:
    """统一目标检测器（封装YOLO + ByteTrack）"""

    def __init__(self, model_yaml: str, model_weights: str, device: str = 'cuda:0',
                 tracker: str = 'bytetrack', channels: int = None, use_simotm: str = None):
        """
        初始化统一检测器

        Args:
            model_yaml: 模型配置文件路径
            model_weights: 模型权重文件路径 (.pt 或 .engine)
            device: 设备 (cuda:0 或 cpu)
            tracker: 跟踪器类型 (bytetrack 或 botsort)
            channels: 输入图像通道数，如果为None则从yaml读取
                - RGB模型: 3
                - IR模型: 3 (红外图像通常也是3通道)
            use_simotm: 预处理方法，如果为None则默认使用'RGB'
                - RGB模型（可见光）: 'RGB'
                - IR模型（红外）: 'SimOTMBBS'

        Examples:
            >>> # RGB 可见光模型
            >>> detector = UnifiedDetector(
            ...     "ultralytics/cfg/models/11/yolo11m.yaml",
            ...     "runs/finetune_V_3classes/weights/last.engine",
            ...     channels=3,
            ...     use_simotm='RGB'
            ... )

            >>> # IR 红外模型
            >>> detector = UnifiedDetector(
            ...     "ultralytics/cfg/models/11/yolo11m.yaml",
            ...     "runs/finetuneNegIR2/weights/last.engine",
            ...     channels=3,
            ...     use_simotm='SimOTMBBS'
            ... )
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
            # 优先使用传入的 channels，否则从 yaml 读取，默认为 3
            self.model_ch = channels if channels is not None else yaml_dict.get('ch', 3)
            # 设置预处理方法
            self.use_simotm = use_simotm if use_simotm is not None else 'RGB'

            # 加载YOLO模型 (支持 .pt 和 .engine)
            self.model = YOLO(model_weights)

            logger.info(f"✓ 模型加载成功")
            logger.info(f"  通道数: {self.model_ch}")
            logger.info(f"  预处理方法: {self.use_simotm}")

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {e}")
            raise

    def detect_and_track(self, frame, conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                        target_size: int = 640) -> List[Dict]:
        """
        检测并跟踪（统一接口）

        自动使用初始化时设置的 channels 和 use_simotm 参数：
        - RGB 模型: channels=3, use_simotm='RGB'
        - IR 模型: channels=3, use_simotm='SimOTMBBS'

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
            # 使用初始化时设置的 use_simotm 和 channels
            results = self.model.track(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=target_size,
                use_simotm=self.use_simotm,  # RGB 或 SimOTMBBS
                channels=self.model_ch,       # 通道数
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
