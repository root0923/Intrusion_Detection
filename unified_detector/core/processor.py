"""
摄像头处理器 - Camera Processor

功能：
- 子进程主逻辑
- 视频流处理
- 统一推理（一次）
- 多规则处理
- 配置热更新
"""
import cv2
import time
import logging
from typing import Dict, Optional
from multiprocessing import Queue
from .detector import UnifiedDetector
from .api_client import APIClient
from ..rules.area_intrusion import AreaIntrusionRule
from ..rules.water_safety import WaterSafetyRule
from ..rules.tripwire import TripwireRule
from ..utils.config_parser import ConfigParser


logger = logging.getLogger(__name__)


class CameraProcessor:
    """摄像头处理器（运行在独立进程中）"""

    def __init__(self, camera_config: Dict, model_yaml: str, model_weights: str,
                 device: str, target_size: int, process_fps: float,
                 api_base_url: str, api_token: str,
                 config_queue: Queue, tracker: str = 'bytetrack'):
        """
        Args:
            camera_config: 摄像头配置（包含三个算法规则）
            model_yaml: 模型配置文件路径
            model_weights: 模型权重文件路径
            device: 设备
            target_size: 推理图像尺寸
            process_fps: 处理帧率
            api_base_url: API基础URL
            api_token: API token
            config_queue: 配置更新队列
            tracker: 跟踪器类型
        """
        self.camera_config = camera_config
        self.camera_key = camera_config['camera_key']
        self.device_name = camera_config['device_name']
        self.channel_name = camera_config['channel_name']

        self.model_yaml = model_yaml
        self.model_weights = model_weights
        self.device = device
        self.target_size = target_size
        self.process_fps = process_fps
        self.tracker = tracker

        self.api_base_url = api_base_url
        self.api_token = api_token
        self.config_queue = config_queue

        # 运行状态
        self.running = False
        self.frame_count = 0

        # 视频流信息
        self.actual_width = None
        self.actual_height = None

        # 组件
        self.detector = None
        self.api_client = None
        self.rules = {}  # {rule_type: RuleEngine实例}
        self.cap = None

        logger.info(f"[{self.camera_key}] CameraProcessor初始化完成")

    def start(self):
        """启动处理循环"""
        try:
            self.running = True

            # 1. 初始化API客户端
            logger.info(f"[{self.camera_key}] 初始化API客户端...")
            self.api_client = APIClient(self.api_base_url, "", "")
            self.api_client.token = self.api_token

            # 2. 获取视频流地址
            logger.info(f"[{self.camera_key}] 获取视频流地址...")
            stream_url = self.api_client.get_stream_url(
                self.camera_config['device_id'],
                self.camera_config['channel_id']
            )

            if not stream_url:
                logger.error(f"[{self.camera_key}] 无法获取视频流地址")
                return

            logger.info(f"[{self.camera_key}] 视频流地址: {stream_url}")

            # 3. 打开视频流
            logger.info(f"[{self.camera_key}] 打开视频流...")
            self.cap = cv2.VideoCapture(stream_url)

            if not self.cap.isOpened():
                logger.error(f"[{self.camera_key}] 无法打开视频流")
                return

            # 获取视频流信息
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps > 120:
                logger.warning(f"[{self.camera_key}] FPS异常({fps})，使用默认值30")
                fps = 30

            logger.info(f"[{self.camera_key}] 视频流信息: {self.actual_width}x{self.actual_height} @ {fps}fps")

            # 4. 坐标转换
            logger.info(f"[{self.camera_key}] 转换坐标...")
            ConfigParser.convert_coordinates(self.camera_config, self.actual_width, self.actual_height)

            # 5. 初始化检测器
            logger.info(f"[{self.camera_key}] 初始化YOLO检测器...")
            self.detector = UnifiedDetector(
                self.model_yaml,
                self.model_weights,
                self.device,
                self.tracker
            )

            # 6. 初始化规则引擎
            logger.info(f"[{self.camera_key}] 初始化规则引擎...")
            self._init_rules()

            # 7. 计算抽帧间隔
            process_interval = max(1, int(round(float(fps) / float(self.process_fps))))
            logger.info(f"[{self.camera_key}] 抽帧设置: 每 {process_interval} 帧处理一次")

            # 8. 主循环
            logger.info(f"[{self.camera_key}] 开始处理循环...")
            self._process_loop(process_interval, fps)

        except Exception as e:
            logger.error(f"[{self.camera_key}] 处理异常: {e}", exc_info=True)

        finally:
            self.stop()

    def _init_rules(self):
        """初始化规则引擎"""
        for rule_type, rule_config in self.camera_config['rules'].items():
            if not rule_config.get('enabled', False):
                continue

            try:
                if rule_type == 'area_intrusion':
                    rule = AreaIntrusionRule(rule_config, self.camera_key)
                elif rule_type == 'tripwire_intrusion':
                    rule = TripwireRule(rule_config, self.camera_key)
                    # 设置图像高度（用于坐标系转换）
                    if self.actual_height is not None:
                        rule.monitor.set_image_height(self.actual_height)
                elif rule_type == 'water_safety':
                    rule = WaterSafetyRule(rule_config, self.camera_key)
                else:
                    logger.warning(f"[{self.camera_key}] 未知规则类型: {rule_type}")
                    continue

                self.rules[rule_type] = rule
                logger.info(f"[{self.camera_key}] ✓ 启用规则: {rule_type}")

            except Exception as e:
                logger.error(f"[{self.camera_key}] 初始化规则失败 [{rule_type}]: {e}", exc_info=True)

    def _process_loop(self, process_interval: int, fps: int):
        """主处理循环"""
        while self.running:
            try:
                # 1. 非阻塞检查配置更新
                self._check_config_update()

                # 2. 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"[{self.camera_key}] 读帧失败")
                    time.sleep(1)
                    continue

                self.frame_count += 1

                # 3. 抽帧检测
                if (self.frame_count - 1) % process_interval == 0:
                    current_time = time.time()

                    # 4. 统一推理（一次）
                    detections = self.detector.detect_and_track(
                        frame,
                        conf_threshold=0.25,  # 统一用0.25，后续规则再过滤
                        iou_threshold=0.7,
                        target_size=self.target_size
                    )

                    # 5. 遍历所有规则处理
                    for rule_type, rule in list(self.rules.items()):
                        try:
                            # 所有规则统一处理（容忍时间在规则内部实现）
                            alarm_info = rule.process(frame, detections, current_time)

                            # 上传报警
                            if alarm_info:
                                rule.trigger_alarm(alarm_info, self.api_client)

                        except Exception as e:
                            logger.error(f"[{self.camera_key}] 规则处理异常 [{rule_type}]: {e}",
                                       exc_info=True)

                # 每5秒打印一次状态
                if self.frame_count % (fps * 5) == 0:
                    logger.debug(f"[{self.camera_key}] 已处理 {self.frame_count} 帧, "
                               f"活跃规则: {list(self.rules.keys())}")

            except Exception as e:
                logger.error(f"[{self.camera_key}] 主循环异常: {e}", exc_info=True)
                time.sleep(0.1)

    def _check_config_update(self):
        """检查配置更新（非阻塞）"""
        try:
            if not self.config_queue.empty():
                new_camera_config = self.config_queue.get_nowait()

                logger.info(f"[{self.camera_key}] 收到配置更新")

                # 更新摄像头配置
                self.camera_config = new_camera_config

                # 更新规则
                self._update_rules()

        except Exception as e:
            logger.error(f"[{self.camera_key}] 配置更新异常: {e}", exc_info=True)

    def _update_rules(self):
        """更新规则（热更新）"""
        new_rules = {}
        old_rule_types = set(self.rules.keys())
        new_rule_types = set(rule_type for rule_type, cfg in self.camera_config['rules'].items()
                            if cfg.get('enabled', False))

        # 删除的规则
        for rule_type in old_rule_types - new_rule_types:
            logger.info(f"[{self.camera_key}] 停用规则: {rule_type}")
            del self.rules[rule_type]

        # 新增或更新的规则
        for rule_type, rule_config in self.camera_config['rules'].items():
            if not rule_config.get('enabled', False):
                continue

            if rule_type in self.rules:
                # 更新现有规则
                self.rules[rule_type].update_config(rule_config)
                new_rules[rule_type] = self.rules[rule_type]
                logger.info(f"[{self.camera_key}] 更新规则: {rule_type}")
            else:
                # 创建新规则
                try:
                    if rule_type == 'area_intrusion':
                        rule = AreaIntrusionRule(rule_config, self.camera_key)
                    elif rule_type == 'tripwire_intrusion':
                        rule = TripwireRule(rule_config, self.camera_key)
                        # 设置图像高度（用于坐标系转换）
                        if self.actual_height is not None:
                            rule.monitor.set_image_height(self.actual_height)
                    elif rule_type == 'water_safety':
                        rule = WaterSafetyRule(rule_config, self.camera_key)
                    else:
                        continue

                    new_rules[rule_type] = rule
                    logger.info(f"[{self.camera_key}] 新增规则: {rule_type}")

                except Exception as e:
                    logger.error(f"[{self.camera_key}] 创建规则失败 [{rule_type}]: {e}", exc_info=True)

        self.rules = new_rules

    def stop(self):
        """停止处理"""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info(f"[{self.camera_key}] 处理器已停止")
