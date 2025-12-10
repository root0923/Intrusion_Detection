"""
绊线入侵检测系统 - 后端API对接版本
支持：多流并行检测、动态配置更新、自动重连、报警上传
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import json
import time
import requests
import base64
import threading
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
from multiprocessing import Process, Event, Queue
import traceback
import torch

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load
from tripwire_intrusion.tripwire_monitor import TripwireMonitor
from tripwire_intrusion.visualizer import TripwireVisualizer
import warnings
warnings.filterwarnings("ignore")


# ============ 配置日志 ============
# 创建log目录
log_dir = Path(__file__).parent / 'log'
log_dir.mkdir(exist_ok=True)

# 日志文件路径（按日期分割）
log_file = log_dir / f"tripwire_{datetime.now().strftime('%Y%m%d')}.log"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # 控制台输出
        logging.StreamHandler(),
        # 文件输出
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============ AES 加密工具 ============
def aes_encrypt_password(password: str) -> str:
    """
    使用AES CBC模式加密密码（与后端Hutool AES加密保持一致）

    Args:
        password: 明文密码

    Returns:
        str: 十六进制编码的加密密文（对应后端encryptHex）
    """
    # AES密钥和IV（固定值，与后端保持一致）
    AES_KEY = b'JzjPLY9632AijnEQ'  # 16字节
    AES_IV = b'DYgjCEIikmj2W9xN'   # 16字节

    try:
        # 创建AES加密器（CBC模式，PKCS7Padding）
        cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)

        # PKCS7填充（AES.block_size = 16）
        padded_data = pad(password.encode('utf-8'), AES.block_size)

        # 加密
        encrypted_data = cipher.encrypt(padded_data)

        # 十六进制编码（对应后端的encryptHex方法）
        encrypted_hex = encrypted_data.hex()

        return encrypted_hex

    except Exception as e:
        logger.error(f"密码加密失败: {e}")
        raise


# ============ API 客户端 ============
class APIClient:
    """后端API客户端"""

    def __init__(self, base_url: str, username: str, password: str):
        """
        Args:
            base_url: 后端基础URL（如 http://localhost:8080）
            username: 用户名
            password: 密码
        """
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()
        self.keep_alive_thread = None
        self.keep_alive_interval = 1200  # 20分钟保活一次
        self.stop_event = threading.Event()

    def login(self) -> bool:
        """
        登录获取token

        Returns:
            bool: 登录是否成功
        """
        try:
            url = f"{self.base_url}/sys/loginToken"

            # AES加密密码
            encrypted_password = aes_encrypt_password(self.password)

            data = {
                "username": self.username,
                "password": encrypted_password
            }

            logger.info(f"正在登录: {url}")
            response = self.session.post(url, json=data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if 'result' in result and 'token' in result['result']:
                    self.token = result['result']['token']
                    logger.info("✓ 登录成功")
                    return True
                else:
                    logger.error(f"✗ 登录失败: 响应中未找到token - {result}")
                    return False
            else:
                logger.error(f"✗ 登录失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"✗ 登录异常: {e}")
            return False

    def keep_alive(self):
        """保活接口（单次调用）"""
        try:
            url = f"{self.base_url}/sys/keepLoginingByToken"
            headers = {"x-access-token": self.token}

            response = self.session.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                logger.debug("✓ 保活成功")
                return True
            else:
                logger.warning(f"✗ 保活失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"✗ 保活异常: {e}")
            return False

    def start_keep_alive(self):
        """启动保活后台线程"""
        def keep_alive_worker():
            logger.info(f"保活线程启动（间隔: {self.keep_alive_interval}s）")
            while not self.stop_event.is_set():
                time.sleep(self.keep_alive_interval)
                if not self.stop_event.is_set():
                    self.keep_alive()

        self.keep_alive_thread = threading.Thread(target=keep_alive_worker, daemon=True)
        self.keep_alive_thread.start()

    def stop_keep_alive(self):
        """停止保活线程"""
        if self.keep_alive_thread:
            self.stop_event.set()
            self.keep_alive_thread.join(timeout=5)
            logger.info("保活线程已停止")

    def get_device_config(self) -> Optional[Dict]:
        """
        获取设备配置

        Returns:
            Dict: 设备配置（包含设备列表、通道、算法规则等）
        """
        try:
            url = f"{self.base_url}/artificial/api/listDeviceAndChannel"
            headers = {"x-access-token": self.token}

            response = self.session.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"✓ 获取设备配置成功")
                return result
            else:
                logger.error(f"✗ 获取设备配置失败: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"✗ 获取设备配置异常: {e}")
            return None

    def get_stream_url(self, device_id: str, channel_id: str) -> Optional[str]:
        """
        获取视频流地址

        Args:
            device_id: 设备ID
            channel_id: 通道ID

        Returns:
            str: RTSP流地址
        """
        try:
            url = f"{self.base_url}/media/api/play/playRealStream"
            headers = {"x-access-token": self.token}
            params = {
                "deviceId": device_id,
                "channelId": channel_id,
                "protocol": "rtsp"
            }

            response = self.session.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 200:
                result = response.json()
                if 'result' in result and 'url' in result['result']:
                    stream_url = result['result']['url']
                    logger.debug(f"✓ 获取流地址成功: {stream_url}")
                    return stream_url
                else:
                    logger.error(f"✗ 获取流地址失败: 响应中未找到url - {result}")
                    return None
            else:
                logger.error(f"✗ 获取流地址失败: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"✗ 获取流地址异常: {e}")
            return None

    def upload_alarm(self, alarm_data: Dict) -> bool:
        """
        上传报警信息

        Args:
            alarm_data: 报警数据

        Returns:
            bool: 上传是否成功
        """
        try:
            url = f"{self.base_url}/artificial/api/alarm"
            headers = {
                "x-access-token": self.token
            }

            response = self.session.post(url, headers=headers, json=alarm_data, timeout=10)

            if response.status_code == 200:
                logger.info("✓ 报警上传成功")
                return True
            else:
                logger.error(f"✗ 报警上传失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"✗ 报警上传异常: {e}")
            return False


# ============ 配置管理器 ============
class ConfigManager:
    """配置管理器：解析设备配置、坐标转换"""

    @staticmethod
    def parse_device_config(config_data: Dict) -> List[Dict]:
        """
        解析设备配置，提取启用且布防的绊线入侵规则

        Args:
            config_data: 从API获取的设备配置

        Returns:
            List[Dict]: 通道配置列表，每个包含：
                - device_id: 设备ID
                - device_name: 设备名称
                - device_code: 设备编码
                - device_ip: 设备IP
                - channel_id: 通道ID
                - channel_name: 通道名称
                - channel_code: 通道编码
                - sensitivity: 置信度阈值
                - repeated_alarm_time: 重复报警间隔
                - direction: 绊线方向
                - frontend_width: 前端显示宽度
                - frontend_height: 前端显示高度
                - is_enable: 是否启用
                - tripwire_points: 绊线点位列表 [[[x1,y1],[x2,y2]], ...]
        """
        channel_configs = []

        try:
            if 'result' not in config_data:
                logger.warning("配置数据中未找到 'result' 字段")
                return channel_configs

            devices = config_data['result']
            if not isinstance(devices, list):
                devices = [devices]

            for device in devices:
                device_id = device.get('deviceId', '')
                device_name = device.get('deviceName', '')
                device_code = device.get('deviceCode', '')
                device_ip = device.get('deviceIp', '')

                # 获取通道列表
                channels = device.get('deviceChannelVos', [])

                for channel in channels:
                    channel_id = channel.get('channelId', '')
                    channel_name = channel.get('channelName', '')
                    channel_code = channel.get('channelCode', '')

                    # 获取算法规则列表
                    algorithm_rules = channel.get('algorithmRules', [])

                    # 筛选绊线入侵规则
                    for rule in algorithm_rules:
                        if rule.get('algorithmCode') != 'tripwire_intrusion':
                            continue

                        # 检查是否启用
                        is_enable = rule.get('izEnable', '0')
                        if is_enable == '0':
                            logger.debug(f"跳过未启用的通道: {device_name}/{channel_name}")
                            continue

                        # 提取配置
                        sensitivity = int(rule.get('sensitivity', 2))
                        mapping = {
                            1: 0.85,  # 最低灵敏度: 非常严格
                            2: 0.75,
                            3: 0.65,
                            4: 0.55,
                            5: 0.45,  # 中等灵敏度
                            6: 0.35,
                            7: 0.25,
                            8: 0.20,
                            9: 0.15,
                            10: 0.10,  # 最高灵敏度: 非常宽松
                        }

                        sensitivity = mapping.get(sensitivity, 0.75)  # 默认0.75
                        repeated_alarm_time = float(rule.get('repeatedAlarmTime', 30.0))
                        direction = rule.get('direction', 'bidirectional')
                        frontend_width = int(rule.get('width', 1920))
                        frontend_height = int(rule.get('height', 1080))

                        # 解析点位列表（polyline）
                        tripwire_points = []
                        algorithm_rule_points = rule.get('algorithmRulePoints', [])
                        for point_item in algorithm_rule_points:
                            if point_item.get('groupType') != 'polyline':
                                continue
                            point_str = point_item.get('pointStr', '')
                            if point_str:
                                try:
                                    # pointStr格式: "[[x1,y1],[x2,y2],[x3,y3],...]"
                                    points = json.loads(point_str)

                                    # 相邻点连线生成绊线：N个点 → N-1条线
                                    # 点0-点1是线1，点1-点2是线2，以此类推
                                    for i in range(len(points) - 1):
                                        line_points = [points[i], points[i + 1]]
                                        tripwire_points.append(line_points)

                                except json.JSONDecodeError as e:
                                    logger.error(f"解析点位失败: {point_str} - {e}")

                        if not tripwire_points:
                            logger.warning(f"通道 {device_name}/{channel_name} 没有有效的点位配置")
                            continue

                        # 创建通道配置
                        channel_config = {
                            'device_id': device_id,
                            'device_name': device_name,
                            'device_code': device_code,
                            'device_ip': device_ip,
                            'channel_id': channel_id,
                            'channel_name': channel_name,
                            'channel_code': channel_code,
                            'sensitivity': sensitivity,
                            'repeated_alarm_time': repeated_alarm_time,
                            'direction': direction,
                            'frontend_width': frontend_width,
                            'frontend_height': frontend_height,
                            'is_enable': is_enable,
                            'tripwire_points': tripwire_points
                        }

                        channel_configs.append(channel_config)
                        logger.info(f"✓ 解析通道配置: {device_name}/{channel_name} "
                                   f"(绊线数: {len(tripwire_points)})")

        except Exception as e:
            logger.error(f"✗ 解析设备配置异常: {e}")
            traceback.print_exc()

        return channel_configs

    @staticmethod
    def convert_tripwire_points(tripwire_points: List[List[List[float]]],
                                frontend_width: int, frontend_height: int,
                                actual_width: int, actual_height: int) -> List[List[List[int]]]:
        """
        将前端坐标转换为实际视频流坐标

        Args:
            tripwire_points: 前端绊线点位 [[[x1,y1],[x2,y2]], ...]
            frontend_width: 前端显示宽度
            frontend_height: 前端显示高度
            actual_width: 实际视频流宽度
            actual_height: 实际视频流高度

        Returns:
            List[List[List[int]]]: 转换后的绊线点位
        """
        scale_x = actual_width / frontend_width
        scale_y = actual_height / frontend_height

        converted = []
        for line_points in tripwire_points:
            converted_line = []
            for point in line_points:
                x, y = point
                actual_x = int(x * scale_x)
                actual_y = int(y * scale_y)
                converted_line.append([actual_x, actual_y])
            converted.append(converted_line)

        logger.debug(f"坐标转换: {frontend_width}x{frontend_height} -> {actual_width}x{actual_height} "
                    f"(scale: {scale_x:.3f}, {scale_y:.3f})")

        return converted

    @staticmethod
    def compare_configs(old_configs: List[Dict], new_configs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        比较新旧配置，找出需要添加、删除、更新的通道

        Args:
            old_configs: 旧配置列表
            new_configs: 新配置列表

        Returns:
            Dict: {'add': [...], 'remove': [...], 'update': [...]}
        """
        # 构建配置字典（以 device_id+channel_id 为key）
        old_dict = {f"{c['device_id']}_{c['channel_id']}": c for c in old_configs}
        new_dict = {f"{c['device_id']}_{c['channel_id']}": c for c in new_configs}

        old_keys = set(old_dict.keys())
        new_keys = set(new_dict.keys())

        # 新增的通道
        added_keys = new_keys - old_keys
        added = [new_dict[k] for k in added_keys]

        # 删除的通道
        removed_keys = old_keys - new_keys
        removed = [old_dict[k] for k in removed_keys]

        # 更新的通道（配置发生变化）
        common_keys = old_keys & new_keys
        updated = []
        for key in common_keys:
            old_cfg = old_dict[key]
            new_cfg = new_dict[key]

            # 比较的字段
            compare_fields = ['sensitivity', 'repeated_alarm_time', 'direction',
                            'frontend_width', 'frontend_height', 'tripwire_points', 'is_enable']

            is_different = False
            for field in compare_fields:
                if old_cfg.get(field) != new_cfg.get(field):
                    is_different = True
                    break

            if is_different:
                updated.append(new_cfg)

        return {
            'add': added,
            'remove': removed,
            'update': updated
        }


# ============ 辅助类：Track对象 ============
class Track:
    """轨迹对象（适配Ultralytics跟踪结果）"""

    def __init__(self, track_id: int, bbox: list, conf: float, cls: int):
        self.track_id = track_id
        self.bbox = bbox
        self.conf = conf
        self.cls = cls
        self.trajectory = deque(maxlen=30)

        # 添加底部中心点到轨迹
        center = self._get_bottom_center(bbox)
        self.trajectory.append(center)

    @staticmethod
    def _get_bottom_center(bbox: list) -> Tuple[float, float]:
        """获取检测框底部中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def get_latest_position(self) -> Tuple[float, float]:
        """获取最新位置"""
        return self.trajectory[-1] if self.trajectory else (0, 0)


# ============ 绊线配置生成器 ============
def generate_tripwire_config(tripwire_points: List[List[List[int]]],
                             channel_code: str,
                             direction: str,
                             cooldown: float) -> Dict:
    """
    动态生成 TripwireMonitor 所需的配置字典

    Args:
        tripwire_points: 绊线点位列表 [[[x1,y1],[x2,y2]], ...]
        channel_code: 通道编码（用于生成绊线ID）
        direction: 绊线方向
        cooldown: 冷却时间

    Returns:
        Dict: 绊线配置字典
    """
    tripwires = []
    for idx, line_points in enumerate(tripwire_points):
        tripwire = {
            "id": f"{channel_code}_line_{idx}",
            "points": line_points,
            "direction": direction,
            "enabled": True,
            "alert_cooldown": cooldown,
            "color": [0, 255, 0]  # 默认绿色
        }
        tripwires.append(tripwire)

    config = {
        "tripwires": tripwires
    }

    return config


# ============ 单通道检测器（进程独立运行）============
def stream_detector_worker(config: Dict, api_base_url: str, api_token: str,
                          model_yaml: str, model_weights: str, device: str,
                          target_size: int, process_fps: float, tracker: str,
                          draw_trajectory: bool, trajectory_length: int,
                          stop_event):
    """
    单个视频流检测进程的工作函数

    Args:
        config: 通道配置
        api_base_url: API基础URL
        api_token: 访问token
        model_yaml: 模型YAML路径
        model_weights: 模型权重路径
        device: 设备
        target_size: YOLO检测目标尺寸
        process_fps: 处理帧率
        tracker: 跟踪器类型
        draw_trajectory: 是否绘制轨迹
        trajectory_length: 轨迹长度
        stop_event: 停止信号
    """
    device_id = config['device_id']
    channel_id = config['channel_id']
    device_name = config['device_name']
    channel_name = config['channel_name']

    logger.info(f"[{device_name}/{channel_name}] 检测进程启动")

    # 获取视频流地址
    api_client = APIClient(api_base_url, "", "")
    api_client.token = api_token
    stream_url = api_client.get_stream_url(device_id, channel_id)

    if not stream_url:
        logger.error(f"[{device_name}/{channel_name}] 无法获取视频流地址")
        return

    logger.info(f"[{device_name}/{channel_name}] 视频流地址: {stream_url}")

    # 初始化YOLO模型
    try:
        logger.info(f"[{device_name}/{channel_name}] 初始化YOLO模型...")
        yaml_dict = yaml_model_load(model_yaml)
        model_ch = yaml_dict.get('ch', 3)

        model = YOLO(model_weights)
        logger.info(f"[{device_name}/{channel_name}] ✓ 模型加载完成 (ch={model_ch}, tracker={tracker})")
    except Exception as e:
        logger.error(f"[{device_name}/{channel_name}] 模型初始化失败: {e}")
        return

    # 轨迹历史
    track_history = {}
    track_last_seen = {}
    max_frames_to_keep = 60

    # 全局报警冷却（通道级别）
    last_alarm_time = None

    # 主检测循环（带自动重连）
    retry_count = 0
    max_retries = 5
    retry_delay = 5
    frame_count = 0

    while not stop_event.is_set() and retry_count < max_retries:
        try:
            # 打开视频流
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                logger.error(f"[{device_name}/{channel_name}] 无法打开视频流")
                retry_count += 1
                time.sleep(retry_delay)
                continue

            # 获取实际视频流尺寸
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps > 120:
                logger.warning(f"FPS异常({fps})，使用默认值30")
                fps = 30

            logger.info(f"[{device_name}/{channel_name}] 视频流信息: {actual_width}x{actual_height} @ {fps}fps")

            # 坐标转换
            converted_points = ConfigManager.convert_tripwire_points(
                config['tripwire_points'],
                config['frontend_width'],
                config['frontend_height'],
                actual_width,
                actual_height
            )

            # 动态生成绊线配置
            tripwire_config = generate_tripwire_config(
                converted_points,
                config['channel_code'],
                config['direction'],
                config['repeated_alarm_time']
            )

            # 初始化 TripwireMonitor（使用临时配置文件）
            temp_config_path = Path(f"temp_tripwire_config_{channel_id}.json")
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(tripwire_config, f, indent=2)

            monitor = TripwireMonitor(
                str(temp_config_path),
                image_height=actual_height,
                global_cooldown=config['repeated_alarm_time']  # 使用全局冷却时间
            )
            visualizer = TripwireVisualizer(
                tripwires=monitor.get_tripwires(),
                draw_trajectory=draw_trajectory,
                trajectory_length=trajectory_length
            )

            logger.info(f"[{device_name}/{channel_name}] ✓ 绊线监控器初始化完成 (绊线数: {len(converted_points)})")

            # 计算抽帧间隔
            process_interval = max(1, int(round(float(fps) / float(process_fps))))
            logger.info(f"[{device_name}/{channel_name}] 抽帧设置: 每 {process_interval} 帧处理一次")

            retry_count = 0  # 重置重试计数
            last_vis_frame = None

            # 帧处理循环
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{device_name}/{channel_name}] 读帧失败，尝试重连...")
                    break

                frame_count += 1

                # 抽帧检测
                if (frame_count - 1) % process_interval == 0:
                    # YOLO检测+跟踪
                    results = model.track(
                        frame,
                        conf=config['sensitivity'],
                        iou=0.7,
                        imgsz=target_size,
                        use_simotm="RGB",
                        channels=3,
                        persist=True,
                        tracker=f"{tracker}.yaml",
                        verbose=False,
                        device=device
                    )

                    # 转换为Track对象
                    current_tracks = []
                    current_track_ids = set()

                    if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        confs = results[0].boxes.conf.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy().astype(int)

                        for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                            current_track_ids.add(track_id)

                            if track_id not in track_history:
                                track_history[track_id] = Track(track_id, box.tolist(), float(conf), int(cls))
                            else:
                                track = track_history[track_id]
                                track.bbox = box.tolist()
                                track.conf = float(conf)
                                track.cls = int(cls)
                                center = track._get_bottom_center(box.tolist())
                                track.trajectory.append(center)

                            track_last_seen[track_id] = frame_count
                            current_tracks.append(track_history[track_id])

                    # 清理旧track
                    tracks_to_remove = []
                    for track_id, last_seen in track_last_seen.items():
                        if frame_count - last_seen > max_frames_to_keep:
                            tracks_to_remove.append(track_id)
                    for track_id in tracks_to_remove:
                        if track_id in track_history:
                            del track_history[track_id]
                        if track_id in track_last_seen:
                            del track_last_seen[track_id]

                    # 绊线监控
                    events = monitor.update(current_tracks)

                    # 可视化
                    class_names = {0: 'person'}
                    vis_frame = visualizer.draw(
                        frame,
                        tracks=current_tracks,
                        recent_events=events,
                        class_names=class_names
                    )
                    last_vis_frame = vis_frame

                    # 处理报警（全局冷却）
                    if events:
                        current_time = time.time()

                        # 检查全局冷却时间
                        if last_alarm_time is None or (current_time - last_alarm_time) >= config['repeated_alarm_time']:
                            # 触发报警
                            event = events[0]  # 取第一个事件

                            # 缩放并编码图片
                            resized_img, img_b64 = resize_and_encode_image(
                                vis_frame,
                                config['frontend_width'],
                                config['frontend_height']
                            )

                            # 上传报警
                            alarm_data = {
                                "deviceId": device_id,
                                "deviceName": device_name,
                                "deviceCode": config['device_code'],
                                "deviceIp": config['device_ip'],
                                "channelId": channel_id,
                                "channelName": channel_name,
                                "channelCode": config['channel_code'],
                                "alarmPicCode": img_b64,
                                "nodeType": "2",
                                "alarmDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "alarmType": "tripwire_intrusion",
                                "alarmTypeName": "绊线入侵"
                            }
                            api_client.upload_alarm(alarm_data)

                            # 更新全局报警时间
                            last_alarm_time = current_time
                            logger.info(f"[{device_name}/{channel_name}] 🚨 报警触发: {event}")

                # 每5秒打印一次状态
                if frame_count % (fps * 5) == 0:
                    logger.debug(f"[{device_name}/{channel_name}] 已处理 {frame_count} 帧")

            cap.release()

            # 清理临时配置文件
            if temp_config_path.exists():
                temp_config_path.unlink()

        except Exception as e:
            logger.error(f"[{device_name}/{channel_name}] 检测异常: {e}")
            traceback.print_exc()
            retry_count += 1
            time.sleep(retry_delay)

    logger.info(f"[{device_name}/{channel_name}] 检测进程退出")


def resize_and_encode_image(image: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, str]:
    """缩放并Base64编码图片"""
    if width and height:
        try:
            resized = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = image
    else:
        resized = image

    success, buffer = cv2.imencode('.jpg', resized)
    if not success:
        success, buffer = cv2.imencode('.jpg', image)

    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return resized, img_b64


# ============ 多流检测管理器 ============
class DetectionManager:
    """多流检测管理器"""

    def __init__(self, api_client: APIClient, model_yaml: str, model_weights: str,
                 device: str = 'cuda:0', target_size: int = 640, process_fps: float = 5.0,
                 tracker: str = 'bytetrack', draw_trajectory: bool = True,
                 trajectory_length: int = 30):
        """
        Args:
            api_client: API客户端
            model_yaml: 模型YAML路径
            model_weights: 模型权重路径
            device: 设备
            target_size: YOLO检测目标尺寸
            process_fps: 处理帧率
            tracker: 跟踪器类型
            draw_trajectory: 是否绘制轨迹
            trajectory_length: 轨迹长度
        """
        self.api_client = api_client
        self.model_yaml = model_yaml
        self.model_weights = model_weights
        self.device = device
        self.target_size = target_size
        self.process_fps = process_fps
        self.tracker = tracker
        self.draw_trajectory = draw_trajectory
        self.trajectory_length = trajectory_length

        # 检测进程字典
        self.detectors = {}

    def start_detector(self, config: Dict):
        """启动单个检测进程"""
        key = (config['device_id'], config['channel_id'])

        if key in self.detectors:
            logger.warning(f"检测器已存在: {config['device_name']}/{config['channel_name']}")
            return

        stop_event = Event()

        process = Process(
            target=stream_detector_worker,
            args=(config, self.api_client.base_url, self.api_client.token,
                 self.model_yaml, self.model_weights, self.device,
                 self.target_size, self.process_fps, self.tracker,
                 self.draw_trajectory, self.trajectory_length, stop_event),
            daemon=True
        )

        process.start()

        self.detectors[key] = {
            'process': process,
            'stop_event': stop_event,
            'config': config
        }

        logger.info(f"✓ 启动检测器: {config['device_name']}/{config['channel_name']}")

    def stop_detector(self, device_id: str, channel_id: str):
        """停止单个检测进程"""
        key = (device_id, channel_id)

        if key not in self.detectors:
            logger.warning(f"检测器不存在: {device_id}/{channel_id}")
            return

        detector = self.detectors[key]
        detector['stop_event'].set()
        detector['process'].join(timeout=5)

        if detector['process'].is_alive():
            detector['process'].terminate()
            logger.warning(f"强制终止检测进程: {device_id}/{channel_id}")

        del self.detectors[key]

        logger.info(f"✓ 停止检测器: {device_id}/{channel_id}")

    def reload_detector(self, config: Dict):
        """重启检测器（配置变更时）"""
        key = (config['device_id'], config['channel_id'])

        logger.info(f"重新加载检测器: {config['device_name']}/{config['channel_name']}")

        if key in self.detectors:
            self.stop_detector(config['device_id'], config['channel_id'])

        time.sleep(1)

        self.start_detector(config)

    def stop_all(self):
        """停止所有检测进程"""
        logger.info("停止所有检测器...")

        keys = list(self.detectors.keys())
        for device_id, channel_id in keys:
            self.stop_detector(device_id, channel_id)

        logger.info("✓ 所有检测器已停止")

    def get_status(self) -> Dict:
        """获取检测器状态"""
        status = {
            'running': sum(1 for d in self.detectors.values() if d['process'].is_alive()),
        }
        return status


# ============ 主程序 ============
def main():
    """主程序"""
    import argparse

    parser = argparse.ArgumentParser(description='绊线入侵检测系统 - API对接版本')

    # API配置
    parser.add_argument('--api-url', type=str, required=True,
                       help='后端API基础URL（如 http://localhost:8080）')
    parser.add_argument('--username', type=str, required=True,
                       help='登录用户名')
    parser.add_argument('--password', type=str, required=True,
                       help='登录密码')

    # 模型配置
    parser.add_argument('--model-yaml', type=str,
                       default="ultralytics/cfg/models/11/yolo11x.yaml",
                       help='模型配置YAML文件')
    parser.add_argument('--weights', type=str,
                       default='data/LLVIP_IF-yolo11x-e300-16-pretrained.pt',
                       help='模型权重文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0 或 cpu)')

    # 检测配置
    parser.add_argument('--target-size', type=int, default=640,
                       help='YOLO检测目标尺寸')
    parser.add_argument('--process-fps', type=float, default=5.0,
                       help='每秒处理帧数（抽帧）')
    parser.add_argument('--config-update-interval', type=int, default=30,
                       help='配置更新间隔（秒）')

    # 跟踪配置
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'botsort'],
                       help='跟踪器类型')

    # 可视化配置
    parser.add_argument('--draw-trajectory', action='store_true', default=True,
                       help='绘制轨迹')
    parser.add_argument('--trajectory-length', type=int, default=30,
                       help='轨迹显示长度')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("绊线入侵检测系统 - API对接版本")
    logger.info("="*60)

    # 1. 登录
    logger.info("\n[1/5] 登录后端系统...")
    api_client = APIClient(args.api_url, args.username, args.password)

    if not api_client.login():
        logger.error("登录失败，程序退出")
        return

    # 2. 启动保活线程
    logger.info("\n[2/5] 启动保活线程...")
    api_client.start_keep_alive()

    # 3. 获取初始配置
    logger.info("\n[3/5] 获取初始配置...")
    config_data = api_client.get_device_config()

    if not config_data:
        logger.error("获取配置失败，程序退出")
        api_client.stop_keep_alive()
        return

    current_configs = ConfigManager.parse_device_config(config_data)
    logger.info(f"✓ 解析配置成功: {len(current_configs)} 个通道")

    # 4. 启动检测管理器
    logger.info("\n[4/5] 启动检测管理器...")
    detection_manager = DetectionManager(
        api_client=api_client,
        model_yaml=args.model_yaml,
        model_weights=args.weights,
        device=args.device,
        target_size=args.target_size,
        process_fps=args.process_fps,
        tracker=args.tracker,
        draw_trajectory=args.draw_trajectory,
        trajectory_length=args.trajectory_length
    )

    # 启动所有启用的通道
    for config in current_configs:
        detection_manager.start_detector(config)

    status = detection_manager.get_status()
    logger.info(f"✓ 检测器状态: {status}")

    # 5. 配置更新循环
    logger.info(f"\n[5/5] 启动配置更新循环（间隔: {args.config_update_interval}s）...")
    logger.info("按 Ctrl+C 退出\n")

    try:
        while True:
            time.sleep(args.config_update_interval)

            logger.info("检查配置更新...")

            # 获取最新配置
            new_config_data = api_client.get_device_config()
            if not new_config_data:
                logger.warning("获取配置失败，跳过本次更新")
                continue

            new_configs = ConfigManager.parse_device_config(new_config_data)

            # 比对配置变化
            changes = ConfigManager.compare_configs(current_configs, new_configs)

            # 处理变化
            if changes['add']:
                logger.info(f"新增通道: {len(changes['add'])}")
                for config in changes['add']:
                    detection_manager.start_detector(config)

            if changes['remove']:
                logger.info(f"删除通道: {len(changes['remove'])}")
                for config in changes['remove']:
                    detection_manager.stop_detector(config['device_id'], config['channel_id'])

            if changes['update']:
                logger.info(f"更新通道: {len(changes['update'])}")
                for config in changes['update']:
                    detection_manager.reload_detector(config)

            if not any(changes.values()):
                logger.debug("配置无变化")

            # 更新当前配置
            current_configs = new_configs

            # 打印状态
            status = detection_manager.get_status()
            logger.info(f"当前状态: {status}")

    except KeyboardInterrupt:
        logger.info("\n\n用户中断，正在退出...")

    except Exception as e:
        logger.error(f"主循环异常: {e}")
        traceback.print_exc()

    finally:
        # 清理
        logger.info("\n清理资源...")
        detection_manager.stop_all()
        api_client.stop_keep_alive()
        logger.info("✓ 程序退出")


if __name__ == '__main__':
    main()
