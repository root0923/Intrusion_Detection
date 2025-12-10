"""
绊线监控核心模块
检测目标轨迹与绊线的相交，判断穿越方向，触发报警事件
"""

import time
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from .geometry import check_line_intersection, compute_crossing_direction


class Tripwire:
    """单条绊线"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 绊线配置字典
                {
                    "id": "line_1",
                    "points": [[x1, y1], [x2, y2]],
                    "direction": "left_to_right" | "right_to_left" | "bidirectional",
                    "enabled": true,
                    "alert_cooldown": 2.0
                }
        """
        self.id = config.get('id', 'unknown')
        self.points = config['points']
        self.p1 = tuple(self.points[0])
        self.p2 = tuple(self.points[1])
        self.direction = config.get('direction', 'bidirectional')
        self.enabled = config.get('enabled', True)
        self.alert_cooldown = config.get('alert_cooldown', 2.0)

        # 颜色配置
        self.color = tuple(config.get('color', [0, 255, 0]))  # 默认绿色

    def is_direction_allowed(self, crossing_direction: str) -> bool:
        """
        检查穿越方向是否符合设定

        Args:
            crossing_direction: 'left_to_right' 或 'right_to_left'

        Returns:
            bool: 是否允许
        """
        if self.direction == 'bidirectional':
            return True
        return self.direction == crossing_direction


class CrossingEvent:
    """穿越事件"""

    def __init__(self, track_id: int, tripwire_id: str,
                 direction: str, timestamp: float,
                 position: Tuple[float, float]):
        """
        Args:
            track_id: 目标ID
            tripwire_id: 绊线ID
            direction: 穿越方向
            timestamp: 时间戳
            position: 穿越位置
        """
        self.track_id = track_id
        self.tripwire_id = tripwire_id
        self.direction = direction
        self.timestamp = timestamp
        self.position = position

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'track_id': self.track_id,
            'tripwire_id': self.tripwire_id,
            'direction': self.direction,
            'timestamp': self.timestamp,
            'time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp)),
            'position': self.position
        }

    def __str__(self):
        time_str = time.strftime('%H:%M:%S', time.localtime(self.timestamp))
        return f"[{time_str}] Track {self.track_id} crossed {self.tripwire_id} ({self.direction})"


class TripwireMonitor:
    """绊线监控器"""

    def __init__(self, config_path: str, max_track_history_age: float = 30.0, image_height: Optional[int] = None, global_cooldown: Optional[float] = None):
        """
        Args:
            config_path: 配置文件路径 (JSON)
            max_track_history_age: 保留track历史记录的最大时间（秒），默认30秒
            image_height: 图像高度，用于坐标系转换（可选）
            global_cooldown: 全局冷却时间（秒），如果提供则覆盖配置文件中的alert_cooldown
        """
        self.config_path = Path(config_path)
        self.tripwires: List[Tripwire] = []
        self.track_last_active: Dict[int, float] = {}  # {track_id: last_active_timestamp}
        self.events: List[CrossingEvent] = []

        # 内存管理参数
        self.max_track_history_age = max_track_history_age

        # 图像高度（用于坐标系转换）
        self.image_height = image_height

        # 全局冷却时间（通道级别）
        self._global_cooldown = global_cooldown  # 如果设置，则覆盖配置文件中的值
        self._global_last_alarm_time = None

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 解析绊线
        tripwires_config = config.get('tripwires', [])
        for tw_config in tripwires_config:
            tripwire = Tripwire(tw_config)
            self.tripwires.append(tripwire)

        print(f"✓ 加载了 {len(self.tripwires)} 条绊线")
        for tw in self.tripwires:
            status = "启用" if tw.enabled else "禁用"
            print(f"  - {tw.id}: {tw.p1} -> {tw.p2}, 方向: {tw.direction}, 状态: {status}")

    def set_image_height(self, height: int):
        """
        设置图像高度（用于坐标系转换）

        Args:
            height: 图像高度
        """
        self.image_height = height
        print(f"✓ 图像高度已设置: {height} (将用于坐标系转换)")

    def update(self, tracks: List[Any]) -> List[CrossingEvent]:
        """
        更新监控状态，检测穿越事件（使用全局冷却机制）

        Args:
            tracks: 活跃轨迹列表（Track对象，需要有trajectory和track_id属性）

        Returns:
            List[CrossingEvent]: 本帧触发的穿越事件（最多1个，全局冷却）
        """
        current_events = []
        current_time = time.time()

        # 获取冷却时间（优先使用 global_cooldown，否则使用配置中的第一条绊线的冷却时间）
        if self._global_cooldown is not None:
            cooldown = self._global_cooldown
        elif self.tripwires:
            cooldown = self.tripwires[0].alert_cooldown
        else:
            cooldown = 2.0

        # 检查全局冷却时间
        if self._global_last_alarm_time is not None:
            if (current_time - self._global_last_alarm_time) < cooldown:
                # 仍在冷却期，不检测任何绊线
                return current_events

        for track in tracks:
            # 更新track最后活跃时间
            self.track_last_active[track.track_id] = current_time

            # 需要至少2个位置点才能判断穿越
            if len(track.trajectory) < 2:
                continue

            # 获取最近的两个位置
            positions = list(track.trajectory)
            track_prev = positions[-2]
            track_curr = positions[-1]

            # 检查每条绊线
            for tripwire in self.tripwires:
                # 检查轨迹段是否与绊线相交
                if check_line_intersection(tripwire.p1, tripwire.p2, track_prev, track_curr):
                    # 计算穿越方向（传入图像高度用于坐标系转换）
                    direction = compute_crossing_direction(
                        tripwire.p1, tripwire.p2, track_prev, track_curr,
                        image_height=self.image_height
                    )

                    if direction is None:
                        continue

                    # 检查方向是否符合设定
                    if not tripwire.is_direction_allowed(direction):
                        continue

                    # 创建穿越事件
                    event = CrossingEvent(
                        track_id=track.track_id,
                        tripwire_id=tripwire.id,
                        direction=direction,
                        timestamp=current_time,
                        position=track_curr
                    )

                    current_events.append(event)
                    self.events.append(event)

                    # 更新全局最后报警时间
                    self._global_last_alarm_time = current_time

                    print(f"🚨 {event}")

                    # 触发一次后立即返回（全局冷却）
                    self._cleanup_old_track_history()
                    return current_events

        # 清理过期的track历史记录
        self._cleanup_old_track_history()

        return current_events

    def _cleanup_old_track_history(self):
        """清理过期的track历史记录，防止内存泄漏"""
        current_time = time.time()
        tracks_to_remove = []

        for track_id, last_active in self.track_last_active.items():
            # 如果超过max_track_history_age秒未活跃，标记删除
            if current_time - last_active > self.max_track_history_age:
                tracks_to_remove.append(track_id)

        # 删除过期track的历史记录
        for track_id in tracks_to_remove:
            if track_id in self.track_last_active:
                del self.track_last_active[track_id]

    def get_tripwires(self) -> List[Tripwire]:
        """获取所有绊线"""
        return self.tripwires

    def get_events(self) -> List[CrossingEvent]:
        """获取所有事件"""
        return self.events

    def export_events(self, output_path: str):
        """
        导出事件到JSON文件

        Args:
            output_path: 输出文件路径
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        events_data = [event.to_dict() for event in self.events]

        with open(output, 'w', encoding='utf-8') as f:
            json.dump({
                'total_events': len(events_data),
                'events': events_data
            }, f, indent=2, ensure_ascii=False)

        print(f"✓ 事件已导出到: {output}")

    def reset(self):
        """重置监控器"""
        self.track_last_active = {}
        self.events = []
        self._global_last_alarm_time = None
