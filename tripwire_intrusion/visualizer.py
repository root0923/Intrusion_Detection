"""
可视化工具
绘制绊线、目标轨迹、检测框、穿越事件等
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .tripwire_monitor import Tripwire, CrossingEvent


class TripwireVisualizer:
    """绊线可视化工具"""

    def __init__(self, tripwires: List[Tripwire],
                 draw_trajectory: bool = True,
                 trajectory_length: int = 30):
        """
        Args:
            tripwires: 绊线列表
            draw_trajectory: 是否绘制轨迹
            trajectory_length: 轨迹显示长度
        """
        self.tripwires = tripwires
        self.draw_trajectory = draw_trajectory
        self.trajectory_length = trajectory_length

        # 颜色配置
        self.colors = self._generate_colors(100)
        self.tripwire_color_active = (0, 255, 0)  # 绿色
        self.tripwire_color_inactive = (128, 128, 128)  # 灰色
        self.event_color = (0, 0, 255)  # 红色

    def draw(self, frame: np.ndarray,
             tracks: List[Any],
             recent_events: List[CrossingEvent] = None,
             class_names: Dict = None) -> np.ndarray:
        """
        在帧上绘制所有可视化元素

        Args:
            frame: 视频帧
            tracks: 目标轨迹列表
            recent_events: 最近的穿越事件（用于高亮显示）
            class_names: 类别名称字典

        Returns:
            绘制后的帧
        """
        vis_frame = frame.copy()

        # 1. 绘制绊线
        self._draw_tripwires(vis_frame)

        # 2. 绘制轨迹
        if self.draw_trajectory:
            self._draw_trajectories(vis_frame, tracks)

        # 3. 绘制检测框和ID
        self._draw_tracks(vis_frame, tracks, class_names)

        # 4. 绘制最近事件
        if recent_events:
            self._draw_recent_events(vis_frame, recent_events)

        # 5. 绘制信息面板
        self._draw_info_panel(vis_frame, len(tracks), len(recent_events) if recent_events else 0)

        return vis_frame

    def _draw_tripwires(self, frame: np.ndarray):
        """绘制绊线"""
        for tripwire in self.tripwires:
            color = self.tripwire_color_active if tripwire.enabled else self.tripwire_color_inactive
            p1 = tuple(map(int, tripwire.p1))
            p2 = tuple(map(int, tripwire.p2))

            # 绘制线段
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

            # 绘制端点
            cv2.circle(frame, p1, 6, color, -1)
            cv2.circle(frame, p2, 6, color, -1)

            # 绘制方向箭头
            if tripwire.enabled and tripwire.direction != 'bidirectional':
                self._draw_direction_arrow(frame, tripwire, color)

            # 绘制标签
            mid_x = int((tripwire.p1[0] + tripwire.p2[0]) / 2)
            mid_y = int((tripwire.p1[1] + tripwire.p2[1]) / 2)
            label = f"{tripwire.id}"

            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (mid_x - label_w // 2 - 5, mid_y - label_h - 10),
                         (mid_x + label_w // 2 + 5, mid_y), color, -1)

            # 标签文字
            cv2.putText(frame, label, (mid_x - label_w // 2, mid_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_direction_arrow(self, frame: np.ndarray, tripwire: Tripwire, color: Tuple):
        """绘制方向箭头"""
        p1 = np.array(tripwire.p1)
        p2 = np.array(tripwire.p2)

        # 计算线段中点
        mid = (p1 + p2) / 2

        # 计算垂直方向
        line_vec = p2 - p1
        perp_vec = np.array([-line_vec[1], line_vec[0]])
        perp_vec = perp_vec / np.linalg.norm(perp_vec) * 20  # 归一化并缩放

        # 根据方向决定箭头位置
        if tripwire.direction == 'left_to_right':
            arrow_end = mid + perp_vec
        else:  # right_to_left
            arrow_end = mid - perp_vec

        cv2.arrowedLine(frame, tuple(map(int, mid)), tuple(map(int, arrow_end)),
                       color, 2, tipLength=0.3)

    def _draw_trajectories(self, frame: np.ndarray, tracks: List[Any]):
        """绘制轨迹"""
        for track in tracks:
            color = self.colors[track.track_id % len(self.colors)]
            trajectory = list(track.trajectory)

            # 绘制轨迹线
            if len(trajectory) >= 2:
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)

                # 绘制轨迹点（渐变透明度效果）
                for i, pt in enumerate(trajectory):
                    alpha = (i + 1) / len(trajectory)  # 从旧到新透明度递增
                    radius = int(2 + alpha * 3)
                    cv2.circle(frame, tuple(map(int, pt)), radius, color, -1, cv2.LINE_AA)

    def _draw_tracks(self, frame: np.ndarray, tracks: List[Any], class_names: Dict = None):
        """绘制检测框和ID"""
        for track in tracks:
            color = self.colors[track.track_id % len(self.colors)]
            x1, y1, x2, y2 = map(int, track.bbox)

            # 绘制框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 标签
            cls_name = class_names.get(track.cls, str(track.cls)) if class_names else str(track.cls)
            label = f"ID:{track.track_id} {cls_name} {track.conf:.2f}"

            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # 标签文字
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_recent_events(self, frame: np.ndarray, events: List[CrossingEvent]):
        """绘制最近的穿越事件"""
        for event in events:
            pos = tuple(map(int, event.position))

            # 绘制警告圆圈
            cv2.circle(frame, pos, 30, self.event_color, 3, cv2.LINE_AA)
            cv2.circle(frame, pos, 35, self.event_color, 2, cv2.LINE_AA)

            # 绘制警告文字
            direction_text = "←" if event.direction == 'right_to_left' else "→"
            cv2.putText(frame, f"ALERT! {direction_text}", (pos[0] - 40, pos[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.event_color, 2)

    def _draw_info_panel(self, frame: np.ndarray, num_tracks: int, num_events: int):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        panel_h = 80
        panel_w = 300

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 文字信息
        y_offset = 35
        cv2.putText(frame, f"Active Tracks: {num_tracks}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 25
        cv2.putText(frame, f"Recent Events: {num_events}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制启用的绊线数量
        enabled_tripwires = sum(1 for tw in self.tripwires if tw.enabled)
        y_offset += 25
        cv2.putText(frame, f"Tripwires: {enabled_tripwires}/{len(self.tripwires)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
        """生成N个不同的颜色"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def draw_tripwire_setup(self, frame: np.ndarray) -> np.ndarray:
        """
        绘制绊线配置图（用于验证配置）

        Args:
            frame: 视频帧

        Returns:
            绘制后的帧
        """
        vis_frame = frame.copy()
        self._draw_tripwires(vis_frame)
        return vis_frame
