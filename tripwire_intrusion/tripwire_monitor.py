"""
绊线监控核心模块
检测目标轨迹与绊线的相交，判断穿越方向，触发报警事件
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from .geometry import check_line_intersection, compute_crossing_direction, compute_point_side
from unified_detector.utils.geometry import is_static_object, is_parallel_movement


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

    def __init__(self, config_path: str, max_track_history_age: float = 30.0, image_height: Optional[int] = None,
                 global_cooldown: Optional[float] = None, first_alarm_time: float = 1.0, tolerance_time: float = 3.0,
                 enable_static_filter: bool = True, enable_parallel_filter: bool = True,
                 static_threshold: float = 15.0, parallel_slope_threshold: float = 0.1):
        """
        Args:
            config_path: 配置文件路径 (JSON)
            max_track_history_age: 保留track历史记录的最大时间（秒），默认30秒
            image_height: 图像高度，用于坐标系转换（可选）
            global_cooldown: 全局冷却时间（秒），如果提供则覆盖配置文件中的alert_cooldown
            first_alarm_time: 首次报警时间（秒），目标持续在危险侧多久后才报警，用于消抖，默认1秒
            tolerance_time: 容忍时间（秒），目标短暂消失后多久重置状态，默认3秒
            enable_static_filter: 是否启用静止过滤
            enable_parallel_filter: 是否启用平行移动过滤
            static_threshold: 静止判断阈值（像素）
            parallel_slope_threshold: 斜率差异阈值
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

        # 过滤规则配置
        self.enable_static_filter = enable_static_filter
        self.enable_parallel_filter = enable_parallel_filter
        self.static_threshold = static_threshold
        self.parallel_slope_threshold = parallel_slope_threshold

        # 全局检测框历史（用于过滤判断）[(timestamp, [bbox1, bbox2, ...]), ...]
        self.detection_history: List[Tuple[float, List[List[float]]]] = []

        # 危险检测历史（记录每次检测到危险的帧）[(timestamp, track_id, position, direction, tripwire_id), ...]
        self.danger_detection_history: List[Tuple[float, int, Tuple[float, float], str, str]] = []

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

    def _should_filter_bbox(self, current_bbox: List[float]) -> bool:
        """
        判断bbox是否应该被过滤（静止或平行移动）

        当历史有2帧时（当前帧是第3帧）：
        1. 静止过滤：检查第1-2帧，删除第1帧中的静止框
        2. 静止过滤：检查第2帧和当前帧，删除第2帧中的静止框，同时判断当前bbox是否静止
        3. 如果删除后历史不足（第1或第2帧为空），删除空帧，返回当前bbox的静止判断结果
        4. 平行过滤：用删除后的第1、2帧和当前帧进行平行判断

        Args:
            current_bbox: 当前bbox

        Returns:
            bool: 是否应该被过滤
        """
        # 需要至少2帧历史才能过滤
        if len(self.detection_history) < 2:
            return False

        # 获取前两帧（会被修改，所以需要拷贝）
        timestamp1, frame1_bboxes = self.detection_history[-2]
        timestamp2, frame2_bboxes = self.detection_history[-1]
        frame1_bboxes = list(frame1_bboxes)
        frame2_bboxes = list(frame2_bboxes)

        # 当前bbox是否被静止过滤
        current_is_static = False

        # === 静止过滤 ===
        if self.enable_static_filter:
            # 1. 检查第1-2帧，删除第1帧中的静止框
            static_in_frame1 = []
            for bbox1 in frame1_bboxes:
                for bbox2 in frame2_bboxes:
                    # 用bbox2作为第2和第3个参数，判断bbox1是否静止
                    if is_static_object(bbox1, bbox2, bbox2, self.static_threshold):
                        static_in_frame1.append(bbox1)
                        print(f"静止过滤：从第1帧删除 {bbox1}")
                        break

            # 从第1帧删除静止框
            frame1_bboxes = [b for b in frame1_bboxes if b not in static_in_frame1]

            # 2. 检查第2帧和当前帧，删除第2帧中的静止框，同时判断当前bbox是否静止
            static_in_frame2 = []
            for bbox2 in frame2_bboxes:
                # 判断bbox2和当前bbox是否静止
                if is_static_object(bbox2, current_bbox, current_bbox, self.static_threshold):
                    static_in_frame2.append(bbox2)
                    current_is_static = True
                    print(f"静止过滤：从第2帧删除 {bbox2}，当前bbox也是静止的")

            # 从第2帧删除静止框
            frame2_bboxes = [b for b in frame2_bboxes if b not in static_in_frame2]

            # 更新历史中的第1、2帧，如果为空则删除
            if len(frame1_bboxes) == 0:
                self.detection_history.pop(-2)
                print(f"第1帧被完全过滤，删除该帧")
            else:
                self.detection_history[-2] = (timestamp1, frame1_bboxes)

            # 更新第2帧（注意：如果第1帧被删除，索引会变化）
            if len(self.detection_history) >= 2:
                if len(frame2_bboxes) == 0:
                    self.detection_history.pop(-1)
                    print(f"第2帧被完全过滤，删除该帧")
                else:
                    self.detection_history[-1] = (timestamp2, frame2_bboxes)
            elif len(self.detection_history) == 1:
                # 第1帧被删除了，第2帧变成了最后一帧
                if len(frame2_bboxes) == 0:
                    self.detection_history.pop(-1)
                    print(f"第2帧被完全过滤，删除该帧")
                else:
                    self.detection_history[-1] = (timestamp2, frame2_bboxes)

        # 如果当前bbox被静止过滤，直接返回True
        if current_is_static:
            return True

        # 检查删除后历史是否足够（历史不足2帧）
        if len(self.detection_history) < 2:
            # 历史不足，无法做平行过滤，不过滤当前帧
            print(f"静止过滤后历史不足，跳过平行过滤")
            return False

        # === 平行过滤 ===
        # 遍历所有可能的组合
        for bbox2 in frame2_bboxes:
            for bbox1 in frame1_bboxes:
                if self.enable_parallel_filter:
                    if is_parallel_movement(bbox1, bbox2, current_bbox, self.parallel_slope_threshold):
                        print(f"平行移动过滤（倒影）: {current_bbox}")
                        return True

        return False

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

        # 1. 收集当前帧所有track的bbox（先收集，最后再加入历史）
        current_bboxes = []
        for track in tracks:
            if hasattr(track, 'bbox'):
                current_bboxes.append(track.bbox)

        # 获取冷却时间（优先使用 global_cooldown，否则使用配置中的第一条绊线的冷却时间）
        if self._global_cooldown is not None:
            cooldown = self._global_cooldown
        elif self.tripwires:
            cooldown = self.tripwires[0].alert_cooldown
        else:
            cooldown = 2.0

        # 检测是否有目标在危险侧
        danger_tracks = []  # 记录所有危险track的信息 [(track_id, position, direction, tripwire_id), ...]
        has_danger_before_filter = False  # 记录过滤前是否有危险目标
        danger_bboxes = []  # 记录未被过滤的危险bbox（用于更新detection_history）

        for track in tracks:
            # 更新track最后活跃时间
            self.track_last_active[track.track_id] = current_time

            # 需要至少1个位置点（当前位置）
            if len(track.trajectory) < 1:
                continue

            # 获取当前位置
            track_curr = track.trajectory[-1]

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
                    has_danger_before_filter = True  # 标记过滤前有危险目标
                    # 检查是否应该被过滤
                    should_filter = self._should_filter_bbox(track.bbox)
                    if should_filter:
                        # 被过滤，不算危险
                        print(f"Track {track.track_id} 被过滤（静止/倒影），不算危险")
                    else:
                        # 记录危险track和bbox
                        danger_tracks.append((track.track_id, track_curr, detected_direction, tripwire.id))
                        danger_bboxes.append(track.bbox)
                        break  # 该track已经被某条绊线检测到，不需要继续检查其他绊线

        # 状态更新和报警逻辑（三帧检测机制）
        if len(danger_tracks) > 0:
            # 当前帧检测到危险，记录所有危险track到历史
            for track_id, position, direction, tripwire_id in danger_tracks:
                self.danger_detection_history.append((
                    current_time, track_id, position, direction, tripwire_id
                ))

            # 更新检测历史（只存储未被过滤的危险bbox）
            self.detection_history.append((current_time, danger_bboxes))

            # 清理超过容忍时间的旧历史
            while len(self.detection_history) > 0:
                if current_time - self.detection_history[0][0] > self.tolerance_time:
                    self.detection_history.pop(0)
                else:
                    break

            # 清理超过容忍时间的旧危险历史
            while len(self.danger_detection_history) > 0:
                if current_time - self.danger_detection_history[0][0] > self.tolerance_time:
                    self.danger_detection_history.pop(0)
                else:
                    break

            # 判断是否报警
            if len(self.danger_detection_history) >= 3:
                first_time = self.danger_detection_history[0][0]
                duration = current_time - first_time

                # 条件1：至少3帧检测 + 持续时间超过首次报警时间
                if duration >= self.first_alarm_time:
                    # 条件2：距离上次报警超过冷却时间
                    if self._global_last_alarm_time is None or (current_time - self._global_last_alarm_time) >= cooldown:
                        # 创建穿越事件（使用最新检测的第一个危险track）
                        track_id, position, direction, tripwire_id = danger_tracks[0]
                        event = CrossingEvent(
                            track_id=track_id,
                            tripwire_id=tripwire_id,
                            direction=direction,
                            timestamp=current_time,
                            position=position
                        )

                        current_events.append(event)
                        self.events.append(event)

                        # 更新全局最后报警时间
                        self._global_last_alarm_time = current_time

                        print(f"🚨 {event} (持续 {duration:.1f}s, 检测帧数: {len(self.danger_detection_history)}, 当前危险数: {len(danger_tracks)})")

                        # 清空危险历史，重新开始
                        self.danger_detection_history.clear()

                        # 触发一次后立即返回（全局冷却）
                        self._cleanup_old_track_history()
                        return current_events
                    else:
                        print(f"⚠️  绊线检测: 冷却中 ({(current_time - self._global_last_alarm_time):.1f}s / {cooldown}s)")
                else:
                    print(f"⚠️  绊线检测: 目标持续在危险侧 ({duration:.1f}s / {self.first_alarm_time}s, 检测帧数: {len(self.danger_detection_history)}/3)")
            else:
                # 历史不足3帧，继续累积
                print(f"⚠️  绊线检测: 危险检测中 (检测帧数: {len(self.danger_detection_history)}/3)")

        elif has_danger_before_filter:
            # 检测到目标在危险侧，但全部被过滤（静止/倒影），清空历史
            print(f"检测到目标在危险侧但全部被过滤（静止/倒影），清空历史")
            self.danger_detection_history.clear()
            self.detection_history.clear()

        else:
            # 当前帧未检测到目标在危险侧
            # 使用容忍时间机制：用第一帧时间判断是否超时
            if len(self.danger_detection_history) > 0:
                first_time = self.danger_detection_history[0][0]
                gap = current_time - first_time
                if gap >= self.tolerance_time:
                    # 超过容忍时间，重置状态
                    duration = current_time - first_time
                    print(f"✓ 绊线检测: 危险解除 (持续 {duration:.1f}s, 容忍时间 {gap:.1f}s 已超过)")
                    self.danger_detection_history.clear()
                    self.detection_history.clear()
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
        self.detection_history = []
        self.danger_detection_history = []