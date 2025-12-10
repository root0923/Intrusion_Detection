"""
区域入侵检测系统 - 后端API对接版本
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
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from multiprocessing import Process, Event, Queue
import traceback

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from detector import Detector
import warnings
warnings.filterwarnings("ignore")


# ============ 配置日志 ============
# 创建log目录
log_dir = Path(__file__).parent / 'log'
log_dir.mkdir(exist_ok=True)

# 日志文件路径（按日期分割）
log_file = log_dir / f"area_intrusion_{datetime.now().strftime('%Y%m%d')}.log"

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
        解析设备配置，提取启用且布防的区域入侵规则

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
                - first_alarm_time: 首次报警时间
                - repeated_alarm_time: 重复报警间隔
                - frontend_width: 前端显示宽度
                - frontend_height: 前端显示高度
                - is_enable: 是否启用
                - point_list: 区域点位列表 [[[x1,y1],[x2,y2],...], ...]
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
                # 如果后期加入设备id过滤条件，可在此处添加
                device_id = device.get('deviceId', '')
                device_name = device.get('deviceName', '')
                device_code = device.get('deviceCode', '')
                device_ip = device.get('deviceIp', '')

                # 获取通道列表
                channels = device.get('deviceChannelVos', [])

                for channel in channels:
                    # 如果后期加入通道id过滤条件，可在此处添加
                    channel_id = channel.get('channelId', '')
                    channel_name = channel.get('channelName', '')
                    channel_code = channel.get('channelCode', '')

                    # 获取算法规则列表
                    algorithm_rules = channel.get('algorithmRules', [])

                    # 筛选区域入侵规则
                    for rule in algorithm_rules:
                        if rule.get('algorithmCode') != 'area_intrusion':
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
                        first_alarm_time = float(rule.get('firstAlarmTime', 1.0))
                        repeated_alarm_time = float(rule.get('repeatedAlarmTime', 30.0))
                        frontend_width = int(rule.get('width', 1920))
                        frontend_height = int(rule.get('height', 1080))

                        # 解析点位列表
                        point_list = []
                        algorithm_rule_points = rule.get('algorithmRulePoints', [])
                        for point_item in algorithm_rule_points:
                            if point_item.get('groupType') != 'polygon':
                                continue
                            point_str = point_item.get('pointStr', '')
                            if point_str:
                                try:
                                    # pointStr格式: "[[x1,y1],[x2,y2],...]" 或 "[[[x1,y1],...],[[x1,y1],...]]"
                                    points = json.loads(point_str)
                                    # 统一转换为三维列表 [region1, region2, ...]
                                    if points and isinstance(points[0][0], list):
                                        # 多区域 [[[x,y],...],[[x,y],...]]
                                        point_list.extend(points)
                                    else:
                                        # 单区域 [[x,y],...]
                                        point_list.append(points)
                                except json.JSONDecodeError as e:
                                    logger.error(f"解析点位失败: {point_str} - {e}")

                        if not point_list:
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
                            'first_alarm_time': first_alarm_time,
                            'repeated_alarm_time': repeated_alarm_time,
                            'frontend_width': frontend_width,
                            'frontend_height': frontend_height,
                            'is_enable': is_enable,
                            'point_list': point_list
                        }

                        channel_configs.append(channel_config)
                        logger.info(f"✓ 解析通道配置: {device_name}/{channel_name} "
                                   f"(区域数: {len(point_list)})")

        except Exception as e:
            logger.error(f"✗ 解析设备配置异常: {e}")
            traceback.print_exc()

        return channel_configs

    @staticmethod
    def convert_points(point_list: List[List[List[float]]],
                      frontend_width: int, frontend_height: int,
                      actual_width: int, actual_height: int) -> List[np.ndarray]:
        """
        将前端坐标转换为实际视频流坐标

        Args:
            point_list: 前端点位列表 [[[x1,y1],[x2,y2],...], ...]
            frontend_width: 前端显示宽度
            frontend_height: 前端显示高度
            actual_width: 实际视频流宽度
            actual_height: 实际视频流高度

        Returns:
            List[np.ndarray]: 转换后的点位列表（每个区域为一个numpy数组）
        """
        scale_x = actual_width / frontend_width
        scale_y = actual_height / frontend_height

        converted_points = []
        for region in point_list:
            region_array = []
            for point in region:
                x, y = point
                actual_x = int(x * scale_x) # 向下取整
                actual_y = int(y * scale_y)
                region_array.append([actual_x, actual_y])
            converted_points.append(np.array(region_array, dtype=np.int32))

        logger.debug(f"坐标转换: {frontend_width}x{frontend_height} -> {actual_width}x{actual_height} "
                    f"(scale: {scale_x:.3f}, {scale_y:.3f})")

        return converted_points

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
        common_keys = old_keys & new_keys # 交集
        updated = []
        for key in common_keys:
            # 简单比较：只比较关键配置字段
            old_cfg = old_dict[key]
            new_cfg = new_dict[key]

            # 比较的字段
            compare_fields = ['sensitivity', 'first_alarm_time', 'repeated_alarm_time',
                            'frontend_width', 'frontend_height', 'point_list', 'is_enable']

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


# ============ ROI管理器（复用现有代码逻辑）============
class ROIManager:
    """ROI区域管理器（使用mask方式）"""

    def __init__(self, roi_points: List[np.ndarray], image_width: int, image_height: int):
        """
        Args:
            roi_points: ROI点位列表（每个为 np.ndarray）
            image_width: 图像宽度
            image_height: 图像高度
        """
        self.image_width = image_width
        self.image_height = image_height
        self.rois = roi_points

        # 创建合并所有ROI的总mask
        self.combined_mask = self._create_combined_mask()

        logger.debug(f"ROI管理器初始化: {len(self.rois)} 个区域")

    def _create_combined_mask(self) -> np.ndarray:
        """创建包含所有ROI的合并mask"""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for roi in self.rois:
            cv2.fillPoly(mask, [roi], 255)
        return mask

    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """将所有ROI区域外的像素变黑"""
        if image.shape[:2] != (self.image_height, self.image_width):
            mask = cv2.resize(self.combined_mask,
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        else:
            mask = self.combined_mask

        masked_image = image.copy()
        masked_image[mask == 0] = 0
        return masked_image

    def draw_rois(self, image: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """在图像上绘制ROI区域"""
        img_draw = image.copy()

        for roi_id, roi in enumerate(self.rois):
            cv2.polylines(img_draw, [roi], isClosed=True, color=color, thickness=thickness)

            # 绘制半透明填充
            overlay = img_draw.copy()
            cv2.fillPoly(overlay, [roi], color)
            cv2.addWeighted(overlay, 0.2, img_draw, 0.8, 0, img_draw)

            # 添加ROI标签
            centroid = roi.mean(axis=0).astype(int)
            cv2.putText(img_draw, f'ROI-{roi_id}', tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img_draw


# ============ 报警管理器（复用现有代码逻辑）============
class AlarmManager:
    """报警管理器"""

    def __init__(self,
                 conf_threshold: float = 0.25,
                 first_alarm_duration: float = 1.0,
                 repeat_alarm_interval: float = 30.0,
                 save_height: Optional[int] = None,
                 save_width: Optional[int] = None):
        """
        Args:
            conf_threshold: 置信度阈值
            first_alarm_duration: 首次报警时间（秒）
            repeat_alarm_interval: 重复报警间隔（秒）
            save_height: 保存报警图片高度
            save_width: 保存报警图片宽度
        """
        self.conf_threshold = conf_threshold
        self.first_alarm_duration = first_alarm_duration
        self.repeat_alarm_interval = repeat_alarm_interval
        self.save_height = save_height
        self.save_width = save_width

        # 入侵状态
        self.intrusion_state = {
            'first_time': None,
            'last_alarm_time': None
        }

    def update_intrusion(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        更新入侵状态并触发报警

        Args:
            detections: 检测结果列表
            frame: 当前帧图像

        Returns:
            List[Dict]: 触发的报警列表
        """
        current_time = time.time()
        alarms = []

        # 过滤置信度
        valid_detections = [det for det in detections if det['conf'] >= self.conf_threshold]

        if valid_detections:
            # 有入侵
            if self.intrusion_state['first_time'] is None:
                # 首次入侵
                self.intrusion_state['first_time'] = current_time
                logger.debug(f"检测到入侵 (消抖中...)")
            else:
                # 持续入侵
                duration = current_time - self.intrusion_state['first_time']

                # 条件1：持续时间超过首次报警时间
                if duration >= self.first_alarm_duration:
                    # 条件2：距离上次报警超过重复报警间隔
                    if (self.intrusion_state['last_alarm_time'] is None or
                        current_time - self.intrusion_state['last_alarm_time'] >= self.repeat_alarm_interval):

                        # 触发报警
                        alarm = self._create_alarm()
                        alarms.append(alarm)

                        # 更新最后报警时间
                        self.intrusion_state['last_alarm_time'] = current_time

                        logger.info(f"🚨 报警触发! (持续 {duration:.1f}s, 检测数: {len(valid_detections)})")
        else:
            # 无入侵
            if self.intrusion_state['first_time'] is not None:
                duration = current_time - self.intrusion_state['first_time']
                logger.debug(f"入侵结束 (持续 {duration:.1f}s)")
                self.intrusion_state['first_time'] = None
                self.intrusion_state['last_alarm_time'] = None

        return alarms

    def _create_alarm(self) -> Dict:
        """创建报警信息"""
        alarm = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return alarm

    def resize_and_encode(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """将图片缩放并Base64编码"""
        img = image

        if self.save_height and self.save_width:
            try:
                resized = cv2.resize(img, (int(self.save_width), int(self.save_height)),
                                   interpolation=cv2.INTER_AREA)
            except Exception:
                resized = img
        else:
            resized = img

        success, buffer = cv2.imencode('.jpg', resized)
        if not success:
            success, buffer = cv2.imencode('.jpg', img)

        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return resized, image_base64


# ============ 单通道检测器（进程独立运行）============
def stream_detector_worker(config: Dict, api_base_url: str, api_token: str,
                          model_yaml: str, model_weights: str, device: str,
                          target_size: int, process_fps: float,
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

    # 初始化检测器
    try:
        detector = Detector(model_yaml, model_weights, device)
    except Exception as e:
        logger.error(f"[{device_name}/{channel_name}] 检测器初始化失败: {e}")
        return

    # 主检测循环（带自动重连）
    retry_count = 0
    max_retries = 5
    retry_delay = 5  # 重连间隔（秒）

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
            if fps <= 0 or fps > 120:  # 异常值检测
                logger.warning(f"FPS异常({fps})，使用默认值30")
                fps = 30

            logger.info(f"[{device_name}/{channel_name}] 视频流信息: {actual_width}x{actual_height} @ {fps}fps")

            # 坐标转换
            converted_points = ConfigManager.convert_points(
                config['point_list'],
                config['frontend_width'],
                config['frontend_height'],
                actual_width,
                actual_height
            )

            # 初始化ROI和报警管理器
            roi_manager = ROIManager(converted_points, actual_width, actual_height)
            alarm_manager = AlarmManager(
                conf_threshold=config['sensitivity'],
                first_alarm_duration=config['first_alarm_time'],
                repeat_alarm_interval=config['repeated_alarm_time'],
                save_height=config['frontend_height'],
                save_width=config['frontend_width']
            )

            # 计算抽帧间隔
            process_interval = max(1, int(round(float(fps) / float(process_fps))))
            logger.info(f"[{device_name}/{channel_name}] 抽帧设置: 每 {process_interval} 帧处理一次")

            frame_count = 0
            retry_count = 0  # 重置重试计数

            # 帧处理循环
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{device_name}/{channel_name}] 读帧失败，尝试重连...")
                    break

                frame_count += 1

                # 抽帧检测
                if (frame_count - 1) % process_interval == 0:
                    # 应用ROI mask
                    masked_frame = roi_manager.apply_mask(frame)

                    # YOLO检测
                    detections = detector.detect(
                        masked_frame,
                        conf_thresh=0.25,
                        iou_thresh=0.7,
                        target_size=target_size
                    )

                    # 更新入侵状态
                    alarms = alarm_manager.update_intrusion(detections, frame)

                    # 处理报警
                    if alarms:
                        # 可视化
                        vis_frame = _visualize_detections(frame, roi_manager, detections,
                                                         alarm_manager, alarms)

                        # 编码图片
                        _, alarm_image_base64 = alarm_manager.resize_and_encode(vis_frame)

                        # 上传报警
                        for alarm in alarms:
                            alarm_data = {
                                "deviceId": device_id,
                                "deviceName": device_name,
                                "deviceCode": config['device_code'],
                                "deviceIp": config['device_ip'],
                                "channelId": channel_id,
                                "channelName": channel_name,
                                "channelCode": config['channel_code'],
                                "alarmPicCode": alarm_image_base64,
                                "nodeType": "2",
                                "alarmDate": alarm['timestamp'],
                                "alarmType": "area_intrusion",
                                "alarmTypeName": "区域入侵"
                            }
                            api_client.upload_alarm(alarm_data)

                # 每5秒打印一次状态
                if frame_count % (fps * 5) == 0:
                    logger.debug(f"[{device_name}/{channel_name}] 已处理 {frame_count} 帧")

            cap.release()

        except Exception as e:
            logger.error(f"[{device_name}/{channel_name}] 检测异常: {e}")
            traceback.print_exc()
            retry_count += 1
            time.sleep(retry_delay)

    logger.info(f"[{device_name}/{channel_name}] 检测进程退出")


def _visualize_detections(frame: np.ndarray, roi_manager: ROIManager,
                         detections: List[Dict], alarm_manager: AlarmManager,
                         alarms: List[Dict]) -> np.ndarray:
    """可视化检测结果"""
    vis_frame = frame.copy()

    # 绘制ROI
    vis_frame = roi_manager.draw_rois(vis_frame, color=(0, 255, 0), thickness=2)

    # 绘制检测框
    class_names = {0: 'person'}
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['conf']
        cls = det['cls']

        is_alarm = conf >= alarm_manager.conf_threshold
        color = (0, 0, 255) if is_alarm else (255, 144, 30)

        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        cls_name = class_names.get(cls, str(cls))
        label = f'{cls_name} {conf:.2f}'

        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 显示报警信息
    if alarms:
        alarm_text = "ALARM! INTRUSION DETECTED"
        cv2.putText(vis_frame, alarm_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return vis_frame


# ============ 多流检测管理器 ============
class DetectionManager:
    """多流检测管理器"""

    def __init__(self, api_client: APIClient, model_yaml: str, model_weights: str,
                 device: str = 'cuda:0', target_size: int = 640, process_fps: float = 10.0):
        """
        Args:
            api_client: API客户端
            model_yaml: 模型YAML路径
            model_weights: 模型权重路径
            device: 设备
            target_size: YOLO检测目标尺寸
            process_fps: 处理帧率
        """
        self.api_client = api_client
        self.model_yaml = model_yaml
        self.model_weights = model_weights
        self.device = device
        self.target_size = target_size
        self.process_fps = process_fps

        # 检测进程字典: {(device_id, channel_id): {'process': ..., 'stop_event': ..., 'config': ...}}
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
                 self.target_size, self.process_fps, stop_event),
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

        time.sleep(1)  # 等待进程完全退出

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
        #     'total': len(self.detectors),
            'running': sum(1 for d in self.detectors.values() if d['process'].is_alive()),
            # 'stopped': sum(1 for d in self.detectors.values() if not d['process'].is_alive()) # 始终为0
        }
        return status


# ============ 主程序 ============
def main():
    """主程序"""
    import argparse

    parser = argparse.ArgumentParser(description='区域入侵检测系统')

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
    parser.add_argument('--process-fps', type=float, default=10.0,
                       help='每秒处理帧数（抽帧）')
    parser.add_argument('--config-update-interval', type=int, default=30,
                       help='配置更新间隔（秒）')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("区域入侵检测系统")
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
        process_fps=args.process_fps
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
