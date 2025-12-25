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
import gc
import logging
import psutil
import os
from typing import Dict, Optional
from multiprocessing import Queue
from .detector import UnifiedDetector
from .api_client import APIClient
from unified_detector.rules.area_intrusion import AreaIntrusionRule
from unified_detector.rules.water_safety import WaterSafetyRule
from unified_detector.rules.tripwire import TripwireRule
from unified_detector.utils.config_parser import ConfigParser
from unified_detector.utils.geometry import draw_detections

logger = logging.getLogger(__name__)


class CameraProcessor:
    """摄像头处理器（运行在独立进程中）"""

    def __init__(self, camera_config: Dict, model_yaml: str, model_weights: str,
                 device: str, target_size: int, process_fps: float,
                 api_base_url: str, api_token: str,
                 config_queue: Queue, tracker: str = 'bytetrack',
                 model_client=None,
                 enable_adaptive_fps: bool = False, fps_idle: float = 1.0,
                 fps_active: float = 5.0, person_timeout: int = 5,
                 tripwire_first_alarm_time: float = 10.0, tripwire_tolerance_time: float = 3.0):
        """
        Args:
            camera_config: 摄像头配置（包含三个算法规则）
            model_yaml: 模型配置文件路径（如果使用model_client则为None）
            model_weights: 模型权重文件路径（如果使用model_client则为None）
            device: 设备（如果使用model_client则为None）
            target_size: 推理图像尺寸
            process_fps: 处理帧率
            api_base_url: API基础URL
            api_token: API token
            config_queue: 配置更新队列
            tracker: 跟踪器类型（如果使用model_client则为None）
            model_client: 轻量级模型客户端（可选，用于多进程共享模型）
            enable_adaptive_fps: 启用自适应帧率
            fps_idle: 无人时的帧率
            fps_active: 有人时的帧率
            person_timeout: 多少秒没检测到人后切换到低帧率
            tripwire_first_alarm_time: 绊线入侵首次报警时间（秒）
            tripwire_tolerance_time: 绊线入侵容忍时间（秒）
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
        self.model_client = model_client  # 新增：轻量级模型客户端

        self.api_base_url = api_base_url
        self.api_token = api_token
        self.config_queue = config_queue

        # 动态帧率配置
        self.enable_adaptive_fps = enable_adaptive_fps
        self.fps_idle = fps_idle
        self.fps_active = fps_active
        self.person_timeout = person_timeout
        self.has_tripwire = False  # 是否有绊线规则（在_init_rules后判断）

        # 绊线入侵配置
        self.tripwire_first_alarm_time = tripwire_first_alarm_time
        self.tripwire_tolerance_time = tripwire_tolerance_time

        # 动态帧率状态
        self.current_fps = process_fps  # 当前使用的帧率（初始为默认值）
        self.last_person_detected_time = 0  # 上次检测到人的时间

        # 运行状态
        self.running = False
        self.frame_count = 0

        # 视频流信息
        self.actual_width = None
        self.actual_height = None

        # 性能统计
        self.perf_stats = {
            'inference_times': [],  # 推理时间
            'rule_times': [],  # 规则处理时间
            'total_times': [],  # 总处理时间
            'processed_frames': 0,
            'frame_intervals': [],  # 帧间隔（实际）
            'last_process_time': None,  # 上次处理时间
        }

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
                self.camera_config['device_code'],
                self.camera_config['channel_code']
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

            # 降低缓冲区大小，防止内存堆积。 保证推理的帧为最新一帧
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"[{self.camera_key}] 已设置缓冲区大小为1")

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
            if self.model_client is not None:
                # 使用共享模型客户端
                logger.info(f"[{self.camera_key}] 使用共享模型服务器...")
                self.detector = self.model_client
                logger.info(f"[{self.camera_key}] ✓ 已连接到模型服务器")
            else:
                # 独立加载模型
                logger.info(f"[{self.camera_key}] 初始化YOLO检测器...")
                self.detector = UnifiedDetector(
                    self.model_yaml,
                    self.model_weights,
                    self.device,
                    self.tracker
                )
                logger.info(f"[{self.camera_key}] ✓ YOLO检测器初始化完成")

            # 6. 初始化规则引擎
            logger.info(f"[{self.camera_key}] 初始化规则引擎...")
            self._init_rules()

            # 判断是否启用动态帧率
            if self.enable_adaptive_fps:
                self.has_tripwire = 'tripwire_intrusion' in self.rules
                if self.has_tripwire:
                    self.current_fps = self.fps_idle  # 启动时使用低帧率
                    logger.info(f"[{self.camera_key}] 启用动态帧率 "
                               f"(空闲:{self.fps_idle}fps, 活跃:{self.fps_active}fps, 超时:{self.person_timeout}s)")
                else:
                    logger.info(f"[{self.camera_key}] 未启用绊线规则，不使用动态帧率")

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
                    rule = TripwireRule(
                        rule_config,
                        self.camera_key,
                        first_alarm_time=self.tripwire_first_alarm_time,
                        tolerance_time=self.tripwire_tolerance_time
                    )
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
        last_gc_time = time.time()  # 记录上次垃圾回收时间
        gc_interval = 30  # 每30秒强制垃圾回收一次
        next_process_time = time.time()  # 下次处理时间戳（基于时间的抽帧）

        while self.running:
            frame = None  # 初始化frame变量
            try:
                # 1. 非阻塞检查配置更新
                self._check_config_update()

                # 2. 先grab帧（不解码，节省内存）
                ret = self.cap.grab()
                if not ret:
                    logger.warning(f"[{self.camera_key}] Grab帧失败")
                    time.sleep(1)
                    continue

                self.frame_count += 1
                current_time = time.time()

                # 3. 基于时间戳的抽帧检测（防止延迟累积）
                if current_time >= next_process_time:
                    # 更新下次处理时间（基于当前时间，防止累积）
                    next_process_time = current_time + (1.0 / self.current_fps)

                    # 只有需要处理时才retrieve（解码帧）
                    ret, frame = self.cap.retrieve()
                    if not ret or frame is None:
                        logger.warning(f"[{self.camera_key}] Retrieve帧失败")
                        continue

                    process_start = time.time()

                    # 记录帧间隔（实际处理间隔）
                    if self.perf_stats['last_process_time'] is not None:
                        frame_interval = (current_time - self.perf_stats['last_process_time']) * 1000  # ms
                        self.perf_stats['frame_intervals'].append(frame_interval)

                        # 理论间隔
                        expected_interval = (1000.0 / self.process_fps)

                        # 延迟检测：实际间隔超过理论间隔50%
                        if frame_interval > expected_interval * 1.5:
                            logger.warning(f"[{self.camera_key}] ⚠️ 帧间隔延迟! "
                                         f"实际: {frame_interval:.1f}ms, "
                                         f"期望: {expected_interval:.1f}ms, "
                                         f"延迟率: {(frame_interval/expected_interval-1)*100:.1f}%")

                    self.perf_stats['last_process_time'] = current_time

                    # 4. 统一推理（一次，不带跟踪）
                    inference_start = time.time()
                    detections = self.detector.detect(
                        frame,
                        conf_threshold=0.25,  # 统一用0.25，后续规则再过滤
                        iou_threshold=0.7,
                        target_size=self.target_size
                    )

                    # 动态帧率：检查是否检测到人（仅对启用绊线规则的进程生效）
                    if self.has_tripwire and self.enable_adaptive_fps:
                        has_person = any(d['cls'] == 0 for d in detections)

                        if has_person:
                            self.last_person_detected_time = current_time

                        # 根据时间窗口决定帧率
                        time_since_person = current_time - self.last_person_detected_time

                        if time_since_person < self.person_timeout:
                            new_fps = self.fps_active
                        else:
                            new_fps = self.fps_idle

                        # 如果帧率变化，记录日志
                        if new_fps != self.current_fps:
                            logger.info(f"[{self.camera_key}] 帧率切换: "
                                       f"{self.current_fps}fps → {new_fps}fps "
                                       f"(最后检测到人: {time_since_person:.1f}秒前)")
                            self.current_fps = new_fps

                    # # 6. 可视化检测结果（调试用）
                    # vis_frame = frame.copy()

                    # # 绘制检测框
                    # vis_frame = draw_detections(vis_frame, detections,
                    #                           conf_threshold=0.25,
                    #                           class_names={0: 'person'})

                    # # 添加摄像头信息
                    # info_text = f"Camera: {self.camera_key} | Frame: {self.frame_count} | Detections: {len(detections)}"
                    # cv2.putText(vis_frame, info_text, (10, 30),
                    #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # vis_frame = cv2.resize(vis_frame, (640, 480))

                    # # 显示图像
                    # cv2.imshow(f'Camera: {self.camera_key}', vis_frame)
                    # cv2.waitKey(1)  # 1ms延迟，允许窗口更新
                    inference_time = (time.time() - inference_start) * 1000  # 转为ms

                    # 5. 遍历所有规则处理
                    rules_start = time.time()
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
                    rules_time = (time.time() - rules_start) * 1000  # 转为ms
                    total_time = (time.time() - process_start) * 1000  # 转为ms

                    # logger.debug(f'规则耗时：{total_time}')

                    # 显式释放检测结果内存
                    del detections

                    # 性能统计
                    self.perf_stats['inference_times'].append(inference_time)
                    self.perf_stats['rule_times'].append(rules_time)
                    self.perf_stats['total_times'].append(total_time)
                    self.perf_stats['processed_frames'] += 1

                    # 保持最近100帧的统计
                    if len(self.perf_stats['inference_times']) > 100:
                        self.perf_stats['inference_times'].pop(0)
                        self.perf_stats['rule_times'].pop(0)
                        self.perf_stats['total_times'].pop(0)
                    if len(self.perf_stats['frame_intervals']) > 100:
                        self.perf_stats['frame_intervals'].pop(0)

                    

                # 每5秒打印一次状态和性能统计
                if self.frame_count % (fps * 5) == 0 and self.perf_stats['processed_frames'] > 0:
                    avg_inference = sum(self.perf_stats['inference_times']) / len(self.perf_stats['inference_times'])
                    avg_rules = sum(self.perf_stats['rule_times']) / len(self.perf_stats['rule_times'])
                    avg_total = sum(self.perf_stats['total_times']) / len(self.perf_stats['total_times'])

                    # 计算帧间隔统计
                    interval_info = ""
                    if len(self.perf_stats['frame_intervals']) > 0:
                        avg_interval = sum(self.perf_stats['frame_intervals']) / len(self.perf_stats['frame_intervals'])
                        expected_interval = 1000.0 / self.process_fps
                        delay_pct = (avg_interval / expected_interval - 1) * 100

                        interval_info = f" | 帧间隔: {avg_interval:.1f}ms (期望: {expected_interval:.1f}ms, 延迟: {delay_pct:+.1f}%)"

                        # 判断是否有延迟
                        if delay_pct > 50:
                            interval_info += " ⚠️ 严重延迟"
                        elif delay_pct > 20:
                            interval_info += " ⚠️ 轻微延迟"
                        elif delay_pct > -10:
                            interval_info += " ✓"

                    logger.info(f"[{self.camera_key}] 帧: {self.frame_count} | "
                               f"当前fps: {self.current_fps:.1f} | "
                               f"推理: {avg_inference:.1f}ms | "
                               f"规则: {avg_rules:.1f}ms | "
                               f"总计: {avg_total:.1f}ms{interval_info} | "
                               f"活跃规则: {list(self.rules.keys())}")

                # 周期性强制垃圾回收
                current_time_gc = time.time()

                # 获取当前内存占用
                process_mem = psutil.Process(os.getpid())
                current_mem_mb = process_mem.memory_info().rss / 1024 / 1024

                if current_time_gc - last_gc_time >= gc_interval:
                    mem_before = current_mem_mb
                    collected = gc.collect()
                    mem_after = process_mem.memory_info().rss / 1024 / 1024
                    freed_mb = mem_before - mem_after

                    if freed_mb > 100:  # 释放超过100MB才记录
                        logger.info(f"[{self.camera_key}] GC完成: 回收{collected}个对象, "
                                  f"释放{freed_mb:.1f}MB ({mem_before:.1f}MB → {mem_after:.1f}MB)")
                    else:
                        logger.debug(f"[{self.camera_key}] GC完成: 回收{collected}个对象")

                    last_gc_time = current_time_gc

            except Exception as e:
                logger.error(f"[{self.camera_key}] 主循环异常: {e}", exc_info=True)
                time.sleep(0.1)

            finally:
                # 显式释放frame对象
                if frame is not None:
                    del frame

    def _check_config_update(self):
        """检查配置更新（非阻塞）"""
        try:
            if not self.config_queue.empty():
                new_camera_config = self.config_queue.get_nowait()

                logger.info(f"[{self.camera_key}] 收到配置更新")

                # 坐标转换
                ConfigParser.convert_coordinates(new_camera_config, self.actual_width, self.actual_height)

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

                # 绊线入侵需要重新设置图像高度（因为reset会重新创建TripwireMonitor）
                if rule_type == 'tripwire_intrusion' and self.actual_height is not None:
                    self.rules[rule_type].monitor.set_image_height(self.actual_height)
                    logger.debug(f"[{self.camera_key}] 绊线入侵规则已重新设置图像高度: {self.actual_height}")

                logger.info(f"[{self.camera_key}] 更新规则: {rule_type}")
            else:
                # 创建新规则
                try:
                    if rule_type == 'area_intrusion':
                        rule = AreaIntrusionRule(rule_config, self.camera_key)
                    elif rule_type == 'tripwire_intrusion':
                        rule = TripwireRule(
                            rule_config,
                            self.camera_key,
                            first_alarm_time=self.tripwire_first_alarm_time,
                            tolerance_time=self.tripwire_tolerance_time
                        )
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

        # 更新动态帧率状态
        if self.enable_adaptive_fps:
            old_has_tripwire = self.has_tripwire
            self.has_tripwire = 'tripwire_intrusion' in self.rules

            if old_has_tripwire != self.has_tripwire:
                if self.has_tripwire:
                    self.current_fps = self.fps_idle
                    self.last_person_detected_time = 0
                    logger.info(f"[{self.camera_key}] 启用动态帧率 "
                               f"(空闲:{self.fps_idle}fps, 活跃:{self.fps_active}fps)")
                else:
                    self.current_fps = self.process_fps
                    logger.info(f"[{self.camera_key}] 禁用动态帧率，恢复固定帧率 {self.process_fps}fps")

    def stop(self):
        """停止处理"""
        self.running = False
        if self.cap:
            self.cap.release()
        # cv2.destroyAllWindows()  # 关闭所有显示窗口
        logger.info(f"[{self.camera_key}] 处理器已停止")