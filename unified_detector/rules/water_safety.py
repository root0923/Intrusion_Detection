"""
涉水安全规则 - Water Safety Rule

功能：
- 检测目标是否进入水域ROI区域
- 支持容忍时间和首次报警时间
- 支持冷却时间
- 逻辑与区域入侵类似，仅报警类型不同
"""
import logging
import numpy as np
from typing import Dict, List, Optional
from .base import RuleEngine
from ..utils.geometry import bbox_center_in_rois, resize_and_encode_image, draw_rois, draw_detections, draw_alarm_text


logger = logging.getLogger(__name__)


class WaterSafetyRule(RuleEngine):
    """涉水安全规则"""

    def _init_rule_specific(self):
        """初始化涉水安全特定配置"""
        # 涉水安全特有配置（与区域入侵相同）
        self.first_alarm_time = self.rule_config.get('first_alarm_time', 1.0)
        self.tolerance_time = 10.0  # 容忍时间，默认3秒
        self.frontend_width = self.rule_config.get('frontend_width', 1920)
        self.frontend_height = self.rule_config.get('frontend_height', 1080)
        self.rois = self.rule_config.get('roi_arrays', [])  # 转换后的ROI坐标

        # 入侵状态管理
        self.intrusion_state = {
            'first_time': None,
            'last_seen_time': None,
        }

        logger.debug(f"[{self.camera_key}] 涉水安全规则初始化: ROI数={len(self.rois)}, "
                    f"sensitivity={self.sensitivity:.2f}, first_alarm_time={self.first_alarm_time}s, "
                    f"tolerance_time={self.tolerance_time}s")

    def process(self, frame, detections: List[Dict], timestamp: float) -> Optional[Dict]:
        """
        处理涉水安全检测

        Args:
            frame: 当前帧图像
            detections: 检测结果列表
            timestamp: 当前时间戳

        Returns:
            alarm_info: 报警信息 or None
        """
        if not self.enabled:
            return None

        # 1. 过滤置信度
        valid_detections = self.filter_by_confidence(detections)

        # 2. 过滤ROI（框中心点在ROI内）
        intruders = []
        for det in valid_detections:
            if bbox_center_in_rois(det['bbox'], self.rois):
                intruders.append(det)

        # 3. 状态更新和报警逻辑
        if len(intruders) > 0:
            # 有涉水行为
            if self.intrusion_state['first_time'] is None:
                # 首次检测到涉水
                self.intrusion_state['first_time'] = timestamp
                self.intrusion_state['last_seen_time'] = timestamp
                logger.debug(f"[{self.camera_key}] 涉水安全: 首次检测到目标 (消抖中...)")
            else:
                # 持续检测到目标
                self.intrusion_state['last_seen_time'] = timestamp

                # 计算持续时间
                duration = timestamp - self.intrusion_state['first_time']

                # 条件1：持续时间超过首次报警时间
                if duration >= self.first_alarm_time:
                    # 条件2：距离上次报警超过冷却时间
                    if self.should_alarm(timestamp):
                        # 触发报警
                        alarm_info = self._create_alarm_info(frame, intruders, duration, timestamp)
                        logger.info(f"[{self.camera_key}] 🚨 涉水安全报警! (持续 {duration:.1f}s, "
                                   f"检测数: {len(intruders)})")
                        return alarm_info

        else:
            # 当前帧未检测到涉水目标
            # 使用容忍时间机制：目标消失后，等待tolerance_time再重置状态
            if self.intrusion_state['first_time'] is not None:
                if self.intrusion_state['last_seen_time'] is not None:
                    gap = timestamp - self.intrusion_state['last_seen_time']
                    if gap >= self.tolerance_time:
                        # 超过容忍时间，重置状态
                        duration = timestamp - self.intrusion_state['first_time']
                        logger.info(f"[{self.camera_key}] 涉水行为结束 (持续 {duration:.1f}s, "
                                   f"容忍时间 {gap:.1f}s 已超过)")
                        self.intrusion_state['first_time'] = None
                        self.intrusion_state['last_seen_time'] = None
                    else: # 容忍时间内，保持状态不变
                        logger.info(f"[{self.camera_key}] 涉水安全检测 暂时未检测到目标 (容忍中: {gap:.1f}s / {self.tolerance_time}s)")

        return None

    def _create_alarm_info(self, frame, intruders: List[Dict], duration: float, timestamp: float) -> Dict:
        """创建报警信息"""
        # 可视化
        vis_frame = frame.copy()

        vis_frame = draw_rois(vis_frame, self.rois, color=(0, 255, 255), thickness=2)

        vis_frame = draw_detections(vis_frame, intruders, conf_threshold=self.sensitivity,
                                    class_names={0: 'person'})

        vis_frame = draw_alarm_text(vis_frame, "ALARM! WATER SAFETY DETECTED")

        # 编码图片
        _, image_base64 = resize_and_encode_image(vis_frame, self.frontend_width, self.frontend_height)

        # 创建报警数据
        alarm_data = self._create_alarm_data(
            alarm_type="water_safety",
            alarm_type_name="涉水检测",
            image_base64=image_base64
        )

        return alarm_data

    def reset(self):
        """重置规则状态"""
        self.intrusion_state = {
            'first_time': None,
            'last_seen_time': None,
        }
        self.last_alarm_time = None
        logger.debug(f"[{self.camera_key}] 涉水安全规则状态已重置")

    def update_config(self, new_config: Dict):
        """更新规则配置（热更新）"""
        self.rule_config = new_config
        self.sensitivity = new_config.get('sensitivity', 0.75)
        self.first_alarm_time = new_config.get('first_alarm_time', 1.0)
        self.repeated_alarm_time = new_config.get('repeated_alarm_time', 30.0)
        self.rois = new_config.get('roi_arrays', [])
        self.device_info = new_config.get('device_info', {})

        # 重置状态
        self.reset()

        logger.info(f"[{self.camera_key}] 涉水安全规则配置已更新")
