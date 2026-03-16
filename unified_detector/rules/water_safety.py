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
from ..utils.geometry import (bbox_center_in_rois, resize_and_encode_image,
                              draw_rois, draw_detections, draw_alarm_text,
                              is_static_object, is_parallel_movement)


logger = logging.getLogger(__name__)


class WaterSafetyRule(RuleEngine):
    """涉水安全规则"""

    def _init_rule_specific(self):
        """初始化涉水安全特定配置"""
        # 涉水安全特有配置（与区域入侵相同）
        self.first_alarm_time = self.rule_config.get('first_alarm_time', 1.0)
        self.tolerance_time = 5.0  # 容忍时间，默认10秒
        self.frontend_width = self.rule_config.get('frontend_width', 1920)
        self.frontend_height = self.rule_config.get('frontend_height', 1080)
        self.rois = self.rule_config.get('roi_arrays', [])  # 转换后的ROI坐标

        # 过滤规则配置
        self.enable_static_filter = self.rule_config.get('enable_static_filter', True)
        self.enable_parallel_filter = self.rule_config.get('enable_parallel_filter', True)
        self.static_threshold = self.rule_config.get('static_threshold', 15.0)  # 静止判断阈值（像素）
        self.parallel_slope_threshold = self.rule_config.get('parallel_slope_threshold', 0.1)  # 斜率差异阈值

        # 检测历史管理（滑动窗口）
        self.detection_history = []  # [(timestamp, bboxes), ...] person检测历史
        self.splash_detection_history = []  # [(timestamp, bboxes), ...] splash检测历史

        # Splash检测配置
        self.person_history_window = 5.0  # person历史记录时间窗口（秒）
        self.splash_history_window = 5.0  # splash历史记录时间窗口（秒）
        self.person_history = []  # [(timestamp, has_person_in_roi), ...] 记录前5秒内是否有person在区域内
        self.splash_history = []  # [(timestamp, has_splash_in_roi), ...] 记录前5秒内是否有splash在区域内

        logger.debug(f"[{self.camera_key}] 涉水安全规则初始化: ROI数={len(self.rois)}, "
                    f"sensitivity={self.sensitivity:.2f}, first_alarm_time={self.first_alarm_time}s, "
                    f"tolerance_time={self.tolerance_time}s, "
                    f"static_filter={self.enable_static_filter}, parallel_filter={self.enable_parallel_filter}, "
                    f"person_history_window={self.person_history_window}s, "
                    f"splash_history_window={self.splash_history_window}s")

    def _apply_filters(self, current_bboxes: List[List[float]]) -> List[List[float]]:
        """
        应用过滤规则，返回未被过滤的检测框

        当历史有2帧时（当前帧是第3帧）：
        1. 静止过滤：检查第1-2帧，删除第1帧中的静止框
        2. 静止过滤：检查第2帧和当前帧，删除第2帧和当前帧中的静止框
        3. 如果删除后历史不足（第1或第2帧为空），删除空帧，返回过滤后的当前帧
        4. 平行过滤：用删除后的第1、2帧和当前帧进行平行判断

        Args:
            current_bboxes: 当前帧的检测框列表

        Returns:
            未被过滤的检测框列表
        """
        if len(self.detection_history) < 2:
            # 历史不足2帧，无法过滤
            return current_bboxes

        # 获取前两帧（会被修改，所以需要拷贝）
        timestamp1, frame1_bboxes = self.detection_history[-2]
        timestamp2, frame2_bboxes = self.detection_history[-1]
        frame1_bboxes = list(frame1_bboxes)
        frame2_bboxes = list(frame2_bboxes)

        # 当前帧的过滤结果（初始为全部）
        current_filtered = list(current_bboxes)

        # === 静止过滤 ===
        if self.enable_static_filter:
            # 1. 检查第1-2帧，删除第1帧中的静止框
            static_in_frame1 = []
            for bbox1 in frame1_bboxes:
                for bbox2 in frame2_bboxes:
                    # 用bbox2作为第2和第3个参数，判断bbox1是否静止
                    if is_static_object(bbox1, bbox2, bbox2, self.static_threshold):
                        static_in_frame1.append(bbox1)
                        logger.debug(f"[{self.camera_key}] 静止过滤：从第1帧删除 {bbox1}")
                        break

            # 从第1帧删除静止框
            frame1_bboxes = [b for b in frame1_bboxes if b not in static_in_frame1]

            # 2. 检查第2帧和当前帧，删除第2帧和当前帧中的静止框
            static_in_frame2 = []
            static_in_current = []
            for bbox2 in frame2_bboxes:
                for bbox3 in current_bboxes:
                    # 判断bbox2和bbox3是否静止
                    if is_static_object(bbox2, bbox3, bbox3, self.static_threshold):
                        if bbox2 not in static_in_frame2:
                            static_in_frame2.append(bbox2)
                            logger.debug(f"[{self.camera_key}] 静止过滤：从第2帧删除 {bbox2}")
                        if bbox3 not in static_in_current:
                            static_in_current.append(bbox3)
                            logger.debug(f"[{self.camera_key}] 静止过滤：从当前帧删除 {bbox3}")

            # 从第2帧和当前帧删除静止框
            frame2_bboxes = [b for b in frame2_bboxes if b not in static_in_frame2]
            current_filtered = [b for b in current_filtered if b not in static_in_current]

            # 更新历史中的第1、2帧，如果为空则删除
            if len(frame1_bboxes) == 0:
                self.detection_history.pop(-2)
                logger.debug(f"[{self.camera_key}] 第1帧被完全过滤，删除该帧")
            else:
                self.detection_history[-2] = (timestamp1, frame1_bboxes)

            # 更新第2帧（注意：如果第1帧被删除，索引会变化）
            if len(self.detection_history) >= 2:
                if len(frame2_bboxes) == 0:
                    self.detection_history.pop(-1)
                    logger.debug(f"[{self.camera_key}] 第2帧被完全过滤，删除该帧")
                else:
                    self.detection_history[-1] = (timestamp2, frame2_bboxes)
            elif len(self.detection_history) == 1:
                # 第1帧被删除了，第2帧变成了最后一帧
                if len(frame2_bboxes) == 0:
                    self.detection_history.pop(-1)
                    logger.debug(f"[{self.camera_key}] 第2帧被完全过滤，删除该帧")
                else:
                    self.detection_history[-1] = (timestamp2, frame2_bboxes)

        # 检查删除后历史是否足够（历史不足2帧）
        if len(self.detection_history) < 2:
            # 历史不足，无法做平行过滤，返回静止过滤后的当前帧
            logger.debug(f"[{self.camera_key}] 静止过滤后历史不足，跳过平行过滤")
            return current_filtered

        # === 平行过滤 ===
        valid_bboxes = []
        for bbox3 in current_filtered:
            filtered = False

            # 遍历所有可能的组合
            for bbox2 in frame2_bboxes:
                for bbox1 in frame1_bboxes:
                    if self.enable_parallel_filter:
                        if is_parallel_movement(bbox1, bbox2, bbox3, self.parallel_slope_threshold):
                            logger.debug(f"[{self.camera_key}] 平行移动过滤（倒影）: {bbox3}")
                            filtered = True
                            break

                if filtered:
                    break

            if not filtered:
                valid_bboxes.append(bbox3)

        return valid_bboxes

    def _apply_filters_splash(self, current_bboxes: List[List[float]]) -> List[List[float]]:
        """
        应用过滤规则，返回未被过滤的splash检测框

        当历史有2帧时（当前帧是第3帧）：
        1. 静止过滤：检查第1-2帧，删除第1帧中的静止框
        2. 静止过滤：检查第2帧和当前帧，删除第2帧和当前帧中的静止框
        3. 如果删除后历史不足（第1或第2帧为空），删除空帧，返回过滤后的当前帧
        4. 平行过滤：用删除后的第1、2帧和当前帧进行平行判断

        Args:
            current_bboxes: 当前帧的检测框列表

        Returns:
            未被过滤的检测框列表
        """
        if len(self.splash_detection_history) < 2:
            # 历史不足2帧，无法过滤
            return current_bboxes

        # 获取前两帧（会被修改，所以需要拷贝）
        timestamp1, frame1_bboxes = self.splash_detection_history[-2]
        timestamp2, frame2_bboxes = self.splash_detection_history[-1]
        frame1_bboxes = list(frame1_bboxes)
        frame2_bboxes = list(frame2_bboxes)

        # 当前帧的过滤结果（初始为全部）
        current_filtered = list(current_bboxes)

        # === 静止过滤 ===
        if self.enable_static_filter:
            # 1. 检查第1-2帧，删除第1帧中的静止框
            static_in_frame1 = []
            for bbox1 in frame1_bboxes:
                for bbox2 in frame2_bboxes:
                    # 用bbox2作为第2和第3个参数，判断bbox1是否静止
                    if is_static_object(bbox1, bbox2, bbox2, self.static_threshold):
                        static_in_frame1.append(bbox1)
                        logger.debug(f"[{self.camera_key}] Splash静止过滤：从第1帧删除 {bbox1}")
                        break

            # 从第1帧删除静止框
            frame1_bboxes = [b for b in frame1_bboxes if b not in static_in_frame1]

            # 2. 检查第2帧和当前帧，删除第2帧和当前帧中的静止框
            static_in_frame2 = []
            static_in_current = []
            for bbox2 in frame2_bboxes:
                for bbox3 in current_bboxes:
                    # 判断bbox2和bbox3是否静止
                    if is_static_object(bbox2, bbox3, bbox3, self.static_threshold):
                        if bbox2 not in static_in_frame2:
                            static_in_frame2.append(bbox2)
                            logger.debug(f"[{self.camera_key}] Splash静止过滤：从第2帧删除 {bbox2}")
                        if bbox3 not in static_in_current:
                            static_in_current.append(bbox3)
                            logger.debug(f"[{self.camera_key}] Splash静止过滤：从当前帧删除 {bbox3}")

            # 从第2帧和当前帧删除静止框
            frame2_bboxes = [b for b in frame2_bboxes if b not in static_in_frame2]
            current_filtered = [b for b in current_filtered if b not in static_in_current]

            # 更新历史中的第1、2帧，如果为空则删除
            if len(frame1_bboxes) == 0:
                self.splash_detection_history.pop(-2)
                logger.debug(f"[{self.camera_key}] Splash第1帧被完全过滤，删除该帧")
            else:
                self.splash_detection_history[-2] = (timestamp1, frame1_bboxes)

            # 更新第2帧（注意：如果第1帧被删除，索引会变化）
            if len(self.splash_detection_history) >= 2:
                if len(frame2_bboxes) == 0:
                    self.splash_detection_history.pop(-1)
                    logger.debug(f"[{self.camera_key}] Splash第2帧被完全过滤，删除该帧")
                else:
                    self.splash_detection_history[-1] = (timestamp2, frame2_bboxes)
            elif len(self.splash_detection_history) == 1:
                # 第1帧被删除了，第2帧变成了最后一帧
                if len(frame2_bboxes) == 0:
                    self.splash_detection_history.pop(-1)
                    logger.debug(f"[{self.camera_key}] Splash第2帧被完全过滤，删除该帧")
                else:
                    self.splash_detection_history[-1] = (timestamp2, frame2_bboxes)

        # 检查删除后历史是否足够（历史不足2帧）
        if len(self.splash_detection_history) < 2:
            # 历史不足，无法做平行过滤，返回静止过滤后的当前帧
            logger.debug(f"[{self.camera_key}] Splash静止过滤后历史不足，跳过平行过滤")
            return current_filtered

        # === 平行过滤 ===
        valid_bboxes = []
        for bbox3 in current_filtered:
            filtered = False

            # 遍历所有可能的组合
            for bbox2 in frame2_bboxes:
                for bbox1 in frame1_bboxes:
                    if self.enable_parallel_filter:
                        if is_parallel_movement(bbox1, bbox2, bbox3, self.parallel_slope_threshold):
                            logger.debug(f"[{self.camera_key}] Splash平行移动过滤（倒影）: {bbox3}")
                            filtered = True
                            break

                if filtered:
                    break

            if not filtered:
                valid_bboxes.append(bbox3)

        return valid_bboxes

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

        # 保存原始检测结果，因为后续会修改detections变量
        original_detections = detections

        # === 新增：Splash检测逻辑 ===
        # 1. 先检查当前帧是否有person（类别0）在区域内，记录到历史
        person_detections = [d for d in original_detections if d.get('cls') == 0 and d['conf'] >= self.sensitivity]
        has_person_in_roi = any(bbox_center_in_rois(d['bbox'], self.rois) for d in person_detections)

        # 记录person历史
        self.person_history.append((timestamp, has_person_in_roi))

        # 清理超过5秒的person历史
        while len(self.person_history) > 0:
            if timestamp - self.person_history[0][0] > self.person_history_window:
                self.person_history.pop(0)
            else:
                break

        # 2. 检查当前帧是否有splash（类别2）在区域内
        # splash使用更低的置信度阈值（0.3）
        splash_detections = [d for d in original_detections if d.get('cls') == 2 and d['conf'] >= 0.45]
        splash_in_roi = [d for d in splash_detections if bbox_center_in_rois(d['bbox'], self.rois)]
        has_splash_in_roi = len(splash_in_roi) > 0

        # 记录splash历史
        self.splash_history.append((timestamp, has_splash_in_roi))

        # 清理超过5秒的splash历史
        while len(self.splash_history) > 0:
            if timestamp - self.splash_history[0][0] > self.splash_history_window:
                self.splash_history.pop(0)
            else:
                break

        # 3. 报警逻辑1：如果当前帧检测到splash且前5秒内有person在区域内，立即报警
        if has_splash_in_roi:
            # 检查前5秒内是否有person在区域内
            has_person_within_5s = any(has_person for _, has_person in self.person_history)

            if has_person_within_5s:
                # 检查冷却时间，避免频繁报警
                if self.should_alarm(timestamp):
                    # 立即报警
                    logger.info(f"[{self.camera_key}] 🚨 检测到splash且前5秒内有person，立即报警!")

                    # 创建报警信息（包含splash的检测）
                    alarm_info = self._create_splash_alarm_info(frame, splash_in_roi)

                    return alarm_info
                else:
                    logger.debug(f"[{self.camera_key}] 检测到splash且前5秒内有person，但在冷却时间内，不报警")

        # 4. 报警逻辑2：如果当前帧检测到person在区域内且前5秒内有splash，立即报警
        if has_person_in_roi:
            # 检查前5秒内是否有splash在区域内
            has_splash_within_5s = any(has_splash for _, has_splash in self.splash_history)

            if has_splash_within_5s:
                # 检查冷却时间，避免频繁报警
                if self.should_alarm(timestamp):
                    # 立即报警
                    logger.info(f"[{self.camera_key}] 🚨 检测到person且前5秒内有splash，立即报警!")

                    # 创建报警信息（包含person的检测）
                    person_in_roi = [d for d in person_detections if bbox_center_in_rois(d['bbox'], self.rois)]
                    alarm_info = self._create_person_alarm_info(frame, person_in_roi)

                    return alarm_info
                else:
                    logger.debug(f"[{self.camera_key}] 检测到person且前5秒内有splash，但在冷却时间内，不报警")

        # === 原有逻辑：涉水安全检测（仅处理person） ===
        # 1. 只处理 person（类别0），过滤掉其他类别
        detections = [d for d in detections if d.get('cls') == 0]

        # 2. 过滤置信度
        valid_detections = self.filter_by_confidence(detections)

        # 3. 过滤ROI（框中心点在ROI内）
        intruders = []
        intruder_bboxes = []
        for det in valid_detections:
            if bbox_center_in_rois(det['bbox'], self.rois):
                intruders.append(det)
                intruder_bboxes.append(det['bbox'])

        # 3. 应用过滤规则（静止/平行移动）
        if len(intruder_bboxes) > 0:
            # 应用过滤（如果历史足够）
            filtered_bboxes = self._apply_filters(intruder_bboxes)

            # 更新检测历史
            if len(filtered_bboxes) > 0:
                # 有未被过滤的框
                self.detection_history.append((timestamp, filtered_bboxes))

                # 清理超过容忍时间的旧历史
                while len(self.detection_history) > 0:
                    if timestamp - self.detection_history[0][0] > self.tolerance_time:
                        self.detection_history.pop(0)
                    else:
                        break

                # 判断是否报警
                if len(self.detection_history) >= 3:
                    first_time = self.detection_history[0][0]
                    duration = timestamp - first_time

                    # 条件1：至少3帧检测 + 持续时间超过首次报警时间
                    if duration >= self.first_alarm_time:
                        # 条件2：距离上次报警超过冷却时间
                        if self.should_alarm(timestamp):
                            # 触发报警（使用过滤后的框重建检测结果）
                            filtered_intruders = [det for det in intruders if det['bbox'] in filtered_bboxes]
                            alarm_info = self._create_alarm_info(frame, filtered_intruders, duration, timestamp)
                            logger.info(f"[{self.camera_key}] 🚨 涉水安全报警! (持续 {duration:.1f}s, "
                                       f"检测帧数: {len(self.detection_history)}, 当前检测数: {len(filtered_intruders)})")

                            # 清空历史，重新开始
                            self.detection_history.clear()

                            return alarm_info

                    logger.debug(f"[{self.camera_key}] 涉水安全检测中 (持续 {duration:.1f}s, "
                               f"检测帧数: {len(self.detection_history)})")
                else:
                    # 历史不足3帧，继续累积
                    logger.debug(f"[{self.camera_key}] 涉水安全检测中 (检测帧数: {len(self.detection_history)}/3)")
            else:
                # 全部被过滤，直接清空历史
                logger.debug(f"[{self.camera_key}] 检测到目标但全部被过滤（静止/倒影），清空历史")
                self.detection_history.clear()
        else:
            # 当前帧未检测到涉水目标
            # 使用容忍时间机制：用第一帧时间判断是否超时
            if len(self.detection_history) > 0:
                first_time = self.detection_history[0][0]
                gap = timestamp - first_time
                if gap >= self.tolerance_time:
                    # 超过容忍时间，重置状态
                    logger.info(f"[{self.camera_key}] 涉水行为结束 (总时长 {gap:.1f}s 已超过容忍时间)")
                    self.detection_history.clear()
                else:
                    # 容忍时间内，保持状态不变
                    logger.debug(f"[{self.camera_key}] 涉水安全检测 暂时未检测到目标 (容忍中: {gap:.1f}s / {self.tolerance_time}s)")

        # === 新增：Splash检测逻辑 ===
        # 1. 只处理 splash（类别2），过滤掉其他类别
        valid_splash_detections = [d for d in original_detections if d.get('cls') == 2 and d['conf'] >= 0.45]

        # 3. 过滤ROI（框底部中心点在ROI内）
        splash_intruders = []
        splash_intruder_bboxes = []
        for det in valid_splash_detections:
            if bbox_center_in_rois(det['bbox'], self.rois):
                splash_intruders.append(det)
                splash_intruder_bboxes.append(det['bbox'])

        # 4. 应用过滤规则（静止/平行移动）
        if len(splash_intruder_bboxes) > 0:
            # 应用过滤（如果历史足够）
            filtered_splash_bboxes = self._apply_filters_splash(splash_intruder_bboxes)

            # 更新检测历史
            if len(filtered_splash_bboxes) > 0:
                # 有未被过滤的框
                self.splash_detection_history.append((timestamp, filtered_splash_bboxes))

                # 清理超过容忍时间的旧历史
                while len(self.splash_detection_history) > 0:
                    if timestamp - self.splash_detection_history[0][0] > self.tolerance_time:
                        self.splash_detection_history.pop(0)
                    else:
                        break

                # 判断是否报警
                if len(self.splash_detection_history) >= 3:
                    first_time = self.splash_detection_history[0][0]
                    duration = timestamp - first_time

                    # 条件1：至少3帧检测 + 持续时间超过首次报警时间
                    if duration >= self.first_alarm_time:
                        # 条件2：距离上次报警超过冷却时间
                        if self.should_alarm(timestamp):
                            # 触发报警（使用过滤后的框重建检测结果）
                            filtered_splash_intruders = [det for det in splash_intruders if det['bbox'] in filtered_splash_bboxes]
                            alarm_info = self._create_splash_filtered_alarm_info(frame, filtered_splash_intruders)
                            logger.info(f"[{self.camera_key}] 🚨 Splash检测报警! (持续 {duration:.1f}s, "
                                       f"检测帧数: {len(self.splash_detection_history)}, 当前检测数: {len(filtered_splash_intruders)})")

                            # 清空历史，重新开始
                            self.splash_detection_history.clear()

                            return alarm_info

                    logger.debug(f"[{self.camera_key}] Splash检测中 (持续 {duration:.1f}s, "
                               f"检测帧数: {len(self.splash_detection_history)})")
                else:
                    # 历史不足3帧，继续累积
                    logger.debug(f"[{self.camera_key}] Splash检测中 (检测帧数: {len(self.splash_detection_history)}/3)")
            else:
                # 全部被过滤，直接清空历史
                logger.debug(f"[{self.camera_key}] 检测到Splash但全部被过滤（静止/倒影），清空历史")
                self.splash_detection_history.clear()
        else:
            # 当前帧未检测到splash目标
            # 使用容忍时间机制：用第一帧时间判断是否超时
            if len(self.splash_detection_history) > 0:
                first_time = self.splash_detection_history[0][0]
                gap = timestamp - first_time
                if gap >= self.tolerance_time:
                    # 超过容忍时间，重置状态
                    logger.info(f"[{self.camera_key}] Splash检测结束 (总时长 {gap:.1f}s 已超过容忍时间)")
                    self.splash_detection_history.clear()
                else:
                    # 容忍时间内，保持状态不变
                    logger.debug(f"[{self.camera_key}] Splash检测 暂时未检测到目标 (容忍中: {gap:.1f}s / {self.tolerance_time}s)")

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

    def _create_splash_alarm_info(self, frame, splash_detections: List[Dict]) -> Dict:
        """创建splash报警信息"""
        # 可视化
        vis_frame = frame.copy()

        # 绘制ROI区域
        vis_frame = draw_rois(vis_frame, self.rois, color=(0, 255, 255), thickness=2)

        # 绘制splash检测框（使用0.25阈值和青色）
        vis_frame = draw_detections(vis_frame, splash_detections,
                                    conf_threshold=self.sensitivity,
                                    class_names={2: 'splash'},
                                    class_conf_thresholds={2: 0.25},
                                    class_colors={2: (255, 255, 0)})

        # 添加报警文本
        vis_frame = draw_alarm_text(vis_frame, "ALARM! SPLASH DETECTED WITH PERSON")

        # 编码图片
        _, image_base64 = resize_and_encode_image(vis_frame, self.frontend_width, self.frontend_height)

        # 创建报警数据（使用涉水检测类型）
        alarm_data = self._create_alarm_data(
            alarm_type="water_safety",
            alarm_type_name="涉水检测",
            image_base64=image_base64
        )

        return alarm_data

    def _create_person_alarm_info(self, frame, person_detections: List[Dict]) -> Dict:
        """创建person报警信息（前5秒内有splash）"""
        # 可视化
        vis_frame = frame.copy()

        # 绘制ROI区域
        vis_frame = draw_rois(vis_frame, self.rois, color=(0, 255, 255), thickness=2)

        # 绘制person检测框
        vis_frame = draw_detections(vis_frame, person_detections, conf_threshold=self.sensitivity,
                                    class_names={0: 'person'})

        # 添加报警文本
        vis_frame = draw_alarm_text(vis_frame, "ALARM! PERSON DETECTED WITH SPLASH")

        # 编码图片
        _, image_base64 = resize_and_encode_image(vis_frame, self.frontend_width, self.frontend_height)

        # 创建报警数据（使用涉水检测类型）
        alarm_data = self._create_alarm_data(
            alarm_type="water_safety",
            alarm_type_name="涉水检测",
            image_base64=image_base64
        )

        return alarm_data

    def _create_splash_filtered_alarm_info(self, frame, splash_detections: List[Dict]) -> Dict:
        """创建splash报警信息（经过静止和平行移动过滤）"""
        # 可视化
        vis_frame = frame.copy()

        # 绘制ROI区域
        vis_frame = draw_rois(vis_frame, self.rois, color=(0, 255, 255), thickness=2)

        # 绘制splash检测框（使用0.25阈值和青色）
        vis_frame = draw_detections(vis_frame, splash_detections,
                                    conf_threshold=self.sensitivity,
                                    class_names={2: 'splash'},
                                    class_conf_thresholds={2: 0.25},
                                    class_colors={2: (255, 255, 0)})

        # 添加报警文本
        vis_frame = draw_alarm_text(vis_frame, "ALARM! SPLASH DETECTED (FILTERED)")

        # 编码图片
        _, image_base64 = resize_and_encode_image(vis_frame, self.frontend_width, self.frontend_height)

        # 创建报警数据（使用涉水检测类型）
        alarm_data = self._create_alarm_data(
            alarm_type="water_safety",
            alarm_type_name="涉水检测",
            image_base64=image_base64
        )

        return alarm_data

    def reset(self):
        """重置规则状态"""
        self.detection_history = []
        self.splash_detection_history = []
        self.person_history = []
        self.splash_history = []
        self.last_alarm_time = None
        logger.debug(f"[{self.camera_key}] 涉水安全规则状态已重置")

    def update_config(self, new_config: Dict):
        """更新规则配置（热更新）"""
        self.rule_config = new_config
        self.sensitivity = new_config.get('sensitivity', 0.75)
        self.first_alarm_time = new_config.get('first_alarm_time', 1.0)
        self.repeated_alarm_time = new_config.get('repeated_alarm_time', 30.0)
        self.tolerance_time = 5.0
        self.rois = new_config.get('roi_arrays', [])
        self.device_info = new_config.get('device_info', {})

        # 更新过滤规则配置
        self.enable_static_filter = new_config.get('enable_static_filter', True)
        self.enable_parallel_filter = new_config.get('enable_parallel_filter', True)
        self.static_threshold = new_config.get('static_threshold', 15.0)
        self.parallel_slope_threshold = new_config.get('parallel_slope_threshold', 0.1)

        # 重置状态
        self.reset()

        logger.info(f"[{self.camera_key}] 涉水安全规则配置已更新")
