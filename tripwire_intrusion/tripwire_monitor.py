"""
绊线监控核心模块
检测目标轨迹与绊线的相交，判断穿越方向，触发报警事件
"""

import time
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from .geometry import check_line_intersection, compute_crossing_direction, compute_point_side


class Tripwire:
    """单条绊线"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 绊线配置字典
                {
                    "id": "line_1",
                    "points": [[x1, y1], [x2, y2]],
                    "direction": "left-to-right" | "right-to-left" | "double-direction",
                    "enabled": true,
                    "alert_cooldown": 2.0
                }
        """
        self.id = config.get('id', 'unknown')
        self.points = config['points']
        self.p1 = tuple(self.points[0])
        self.p2 = tuple(self.points[1])
        self.direction = config.get('direction', 'double-direction')
        self.enabled = config.get('enabled', True)
        self.alert_cooldown = config.get('alert_cooldown', 2.0)

        # 颜色配置
        self.color = tuple(config.get('color', [0, 255, 0]))  # 默认绿色

    def is_direction_allowed(self, crossing_direction: str) -> bool:
        """
        检查穿越方向是否符合设定

        Args:
            crossing_direction: 'left-to-right' 或 'right-to-left'

        Returns:
            bool: 是否允许
        """
        if self.direction == 'double-direction':
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

    def __init__(self, config_path: str, max_track_history_age: float = 30.0, image_height: Optional[int] = None, global_cooldown: Optional[float] = None, first_alarm_time: float = 1.0, tolerance_time: float = 3.0):
        """
        Args:
            config_path: 配置文件路径 (JSON)
            max_track_history_age: 保留track历史记录的最大时间（秒），默认30秒
            image_height: 图像高度，用于坐标系转换（可选）
            global_cooldown: 全局冷却时间（秒），如果提供则覆盖配置文件中的alert_cooldown
            first_alarm_time: 首次报警时间（秒），目标持续在危险侧多久后才报警，用于消抖，默认1秒
            tolerance_time: 容忍时间（秒），目标短暂消失后多久重置状态，默认3秒
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

        # 首次报警时间和容忍时间（和区域入侵逻辑一致）
        self.first_alarm_time = first_alarm_time
        self.tolerance_time = tolerance_time

        # 危险状态管理（全局状态，不区分具体track）
        self.danger_state = {
            'first_time': None,        # 首次检测到目标在危险侧的时间
            'last_seen_time': None,    # 最后一次看到目标在危险侧的时间（用于容忍时间判断）
        }

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
        更新监控状态，检测目标在危险侧（使用位置检测，支持首次报警时间和容忍时间）

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

        # 检测是否有目标在危险侧
        has_danger = False
        danger_track_id = None
        danger_position = None
        danger_direction = None
        danger_tripwire_id = None

        for track in tracks:
            # 更新track最后活跃时间
            self.track_last_active[track.track_id] = current_time

            # 需要至少1个位置点（当前位置）
            if len(track.trajectory) < 1:
                continue

            # 获取当前位置
            track_curr = track.trajectory[-1]
            print(f'位置：{track_curr}')

            # 检查每条绊线
            for tripwire in self.tripwires:
                if not tripwire.enabled:
                    continue

                # 判断当前位置在绊线的哪一侧
                side = compute_point_side(
                    tripwire.p1, tripwire.p2, track_curr,
                    image_height=self.image_height
                )

                # 如果点在线段延长线上（不在线段范围内），跳过此绊线
                if side == 'outside':
                    continue

                # 根据方向配置判断是否在危险侧
                # left-to-right: 检测右侧
                # right-to-left: 检测左侧
                is_danger = False
                detected_direction = None

                print(tripwire.direction, side)

                if tripwire.direction == 'left-to-right' and side == 'right':
                    is_danger = True
                    detected_direction = 'left-to-right'
                elif tripwire.direction == 'right-to-left' and side == 'left':
                    is_danger = True
                    detected_direction = 'right-to-left'
                elif tripwire.direction == 'double-direction' and side in ['left', 'right']:
                    # 双向检测：任一侧都算危险
                    is_danger = True
                    detected_direction = 'left-to-right' if side == 'right' else 'right-to-left'

                if is_danger:
                    has_danger = True
                    danger_track_id = track.track_id
                    danger_position = track_curr
                    danger_direction = detected_direction
                    danger_tripwire_id = tripwire.id
                    break  # 只要有一条绊线检测到危险就够了

            if has_danger:
                break  # 找到危险目标，不需要继续检查其他track

        # 状态更新和报警逻辑（和区域入侵一致）
        if has_danger:
            # 有目标在危险侧
            if self.danger_state['first_time'] is None:
                # 首次检测到目标在危险侧
                self.danger_state['first_time'] = current_time
                self.danger_state['last_seen_time'] = current_time
                print(f"⚠️  绊线检测: 首次检测到目标在危险侧 (消抖中... 需持续{self.first_alarm_time}秒)")
            else:
                # 持续检测到目标在危险侧，更新最后看到时间
                self.danger_state['last_seen_time'] = current_time

                # 计算持续时间
                duration = current_time - self.danger_state['first_time']

                # 条件：持续时间超过首次报警时间
                if duration >= self.first_alarm_time:
                    # 创建穿越事件
                    event = CrossingEvent(
                        track_id=danger_track_id,
                        tripwire_id=danger_tripwire_id,
                        direction=danger_direction,
                        timestamp=current_time,
                        position=danger_position
                    )

                    current_events.append(event)
                    self.events.append(event)

                    # 更新全局最后报警时间
                    self._global_last_alarm_time = current_time

                    print(f"🚨 {event} (持续 {duration:.1f}s)")

                    # 触发一次后立即返回（全局冷却）
                    self._cleanup_old_track_history()
                    return current_events
                else:
                    print(f"⚠️  绊线检测: 目标持续在危险侧 ({duration:.1f}s / {self.first_alarm_time}s)")

        else:
            # 当前帧未检测到目标在危险侧
            # 使用容忍时间机制：目标消失后，等待tolerance_time再重置状态
            if self.danger_state['first_time'] is not None:
                if self.danger_state['last_seen_time'] is not None:
                    gap = current_time - self.danger_state['last_seen_time']
                    if gap >= self.tolerance_time:
                        # 超过容忍时间，重置状态
                        duration = current_time - self.danger_state['first_time']
                        print(f"✓ 绊线检测: 危险解除 (持续 {duration:.1f}s, 容忍时间 {gap:.1f}s 已超过)")
                        self.danger_state['first_time'] = None
                        self.danger_state['last_seen_time'] = None
                    else:
                        # 容忍时间内，保持状态不变
                        print(f"⚠️  绊线检测: 暂时未检测到目标在危险侧 (容忍中: {gap:.1f}s / {self.tolerance_time}s)")

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
        self.danger_state = {
            'first_time': None,
            'last_seen_time': None,
        }