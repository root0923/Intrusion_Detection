"""
绊线入侵规则 - Tripwire Intrusion Rule

功能：
- 检测目标是否穿越绊线
- 复用TripwireMonitor实现越线判断
- 支持冷却时间
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
from .base import RuleEngine
from ..utils.geometry import resize_and_encode_image, draw_tripwires, draw_detections, draw_alarm_text
from tripwire_intrusion.tripwire_monitor import TripwireMonitor


logger = logging.getLogger(__name__)


class Track:
    """轻量级Track对象（仅包含当前位置，适配TripwireMonitor接口）"""

    def __init__(self, track_id: int, bbox: list, conf: float, cls: int):
        self.track_id = track_id
        self.bbox = bbox
        self.conf = conf
        self.cls = cls
        self.trajectory = deque(maxlen=1)  # 只保留当前位置

        # 添加底部中心点到轨迹
        center = self._get_bottom_center(bbox)
        self.trajectory.append(center)

    @staticmethod
    def _get_bottom_center(bbox: list):
        """获取检测框底部中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)


class TripwireRule(RuleEngine):
    """绊线入侵规则"""

    def __init__(self, rule_config: Dict, camera_key: str,
                 first_alarm_time: float = 10.0, tolerance_time: float = 3.0):
        """
        Args:
            rule_config: 规则配置字典
            camera_key: 摄像头唯一标识
            first_alarm_time: 首次报警时间（秒），从main.py参数传入
            tolerance_time: 容忍时间（秒），从main.py参数传入
        """
        # 保存参数供后续使用
        self._first_alarm_time = first_alarm_time
        self._tolerance_time = tolerance_time

        # 调用父类初始化
        super().__init__(rule_config, camera_key)

    def _init_rule_specific(self):
        """初始化绊线入侵特定配置"""
        # 绊线特有配置
        self.direction = self.rule_config.get('direction', 'double-direction')
        self.frontend_width = self.rule_config.get('frontend_width', 1920)
        self.frontend_height = self.rule_config.get('frontend_height', 1080)
        self.tripwire_lines = self.rule_config.get('tripwire_arrays', [])  # 转换后的绊线坐标

        # 使用从main.py传入的参数，而不是从rule_config读取
        self.first_alarm_time = self._first_alarm_time
        self.tolerance_time = self._tolerance_time

        # 创建临时配置文件并初始化TripwireMonitor
        self._init_tripwire_monitor()

        logger.debug(f"[{self.camera_key}] 绊线入侵规则初始化: 绊线数={len(self.tripwire_lines)}, "
                    f"sensitivity={self.sensitivity:.2f}, direction={self.direction}, "
                    f"first_alarm_time={self.first_alarm_time}s, tolerance_time={self.tolerance_time}s")

    def _init_tripwire_monitor(self):
        """初始化TripwireMonitor"""
        # 生成临时配置文件
        temp_config = {
            "tripwires": []
        }

        for idx, line_points in enumerate(self.tripwire_lines):
            tripwire = {
                "id": f"{self.camera_key}_line_{idx}",
                "points": line_points,
                "direction": self.direction,
                "enabled": True,
                "alert_cooldown": self.repeated_alarm_time,
                "color": [0, 255, 0]
            }
            temp_config["tripwires"].append(tripwire)

        # 保存临时配置文件
        self.temp_config_path = Path(f"unified_detector/temp_tripwire_{self.camera_key}.json")
        with open(self.temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(temp_config, f, indent=2)

        # 初始化TripwireMonitor（使用全局冷却时间、首次报警时间和容忍时间）
        self.monitor = TripwireMonitor(
            str(self.temp_config_path),
            global_cooldown=self.repeated_alarm_time,
            first_alarm_time=self.first_alarm_time,
            tolerance_time=self.tolerance_time
        )

        logger.debug(f"[{self.camera_key}] TripwireMonitor初始化完成")

    def process(self, frame, detections: List[Dict], timestamp: float) -> Optional[Dict]:
        """
        处理绊线入侵检测

        Args:
            frame: 当前帧图像
            detections: 检测结果列表（不需要track_id）
            timestamp: 当前时间戳

        Returns:
            alarm_info: 报警信息 or None
        """
        if not self.enabled:
            return None

        # 1. 过滤置信度
        valid_detections = self.filter_by_confidence(detections)

        # 2. 为每个检测框创建临时Track对象（只包含当前位置）
        current_tracks = []
        for idx, det in enumerate(valid_detections):
            # 使用检测框索引作为临时track_id
            temp_track_id = idx
            track = Track(
                temp_track_id, det['bbox'], det['conf'], det['cls']
            )
            current_tracks.append(track)

        # 3. 调用TripwireMonitor检测越线
        events = self.monitor.update(current_tracks)

        # 4. 处理报警（TripwireMonitor内部已处理全局冷却）
        if events:
            event = events[0]  # 取第一个事件
            alarm_info = self._create_alarm_info(frame, event, current_tracks, timestamp)
            logger.info(f"[{self.camera_key}] 🚨 绊线入侵报警! "
                       f"crossed {event.tripwire_id} ({event.direction})")
            return alarm_info

        return None

    def _create_alarm_info(self, frame, event, tracks: List[Track], timestamp: float) -> Dict:
        """创建报警信息"""
        # 可视化
        vis_frame = frame.copy()

        # 绘制绊线
        vis_frame = draw_tripwires(vis_frame, self.tripwire_lines, color=(0, 255, 255), thickness=3)

        # 绘制检测框
        detections_for_draw = [
            {
                'bbox': track.bbox,
                'conf': track.conf,
                'cls': track.cls
            }
            for track in tracks
        ]
        vis_frame = draw_detections(vis_frame, detections_for_draw, conf_threshold=self.sensitivity,
                                    class_names={0: 'person'})

        # 绘制报警文字
        vis_frame = draw_alarm_text(vis_frame, "ALARM! TRIPWIRE CROSSED")

        # 编码图片
        _, image_base64 = resize_and_encode_image(vis_frame, self.frontend_width, self.frontend_height)

        # 创建报警数据
        alarm_data = self._create_alarm_data(
            alarm_type="tripwire_intrusion",
            alarm_type_name="绊线检测",
            image_base64=image_base64
        )

        return alarm_data

    def reset(self):
        """重置规则状态"""
        self.last_alarm_time = None

        # 重新初始化TripwireMonitor
        self._cleanup_temp_files()
        self._init_tripwire_monitor()

        logger.debug(f"[{self.camera_key}] 绊线入侵规则状态已重置")

    def update_config(self, new_config: Dict):
        """更新规则配置（热更新）"""
        self.rule_config = new_config
        self.sensitivity = new_config.get('sensitivity', 0.75)
        self.repeated_alarm_time = new_config.get('repeated_alarm_time', 30.0)
        # first_alarm_time 和 tolerance_time 使用从 main.py 传入的参数，不从配置更新
        self.direction = new_config.get('direction', 'double-direction')
        self.tripwire_lines = new_config.get('tripwire_arrays', [])
        self.device_info = new_config.get('device_info', {})

        # 重置状态
        self.reset()

        logger.info(f"[{self.camera_key}] 绊线入侵规则配置已更新")

    def _cleanup_temp_files(self):
        """清理临时配置文件"""
        if hasattr(self, 'temp_config_path') and self.temp_config_path.exists():
            try:
                self.temp_config_path.unlink()
                logger.debug(f"[{self.camera_key}] 临时配置文件已删除: {self.temp_config_path}")
            except Exception as e:
                logger.warning(f"[{self.camera_key}] 删除临时配置文件失败: {e}")

    def __del__(self):
        """析构函数：清理临时文件"""
        self._cleanup_temp_files()