"""
规则引擎抽象基类 - Rule Engine Base Class

定义规则引擎的通用接口和行为
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class RuleEngine(ABC):
    """规则引擎抽象基类"""

    def __init__(self, rule_config: Dict, camera_key: str):
        """
        Args:
            rule_config: 规则配置字典
            camera_key: 摄像头唯一标识
        """
        self.camera_key = camera_key
        self.rule_config = rule_config
        self.enabled = rule_config.get('enabled', True)

        # 通用配置
        self.sensitivity = rule_config.get('sensitivity', 0.75)
        self.repeated_alarm_time = rule_config.get('repeated_alarm_time', 30.0)
        self.device_info = rule_config.get('device_info', {})

        # 报警状态管理
        self.last_alarm_time = None  # 上次报警时间（用于冷却期判断）

        # 规则特定的初始化
        self._init_rule_specific()

    @abstractmethod
    def _init_rule_specific(self):
        """规则特定的初始化（子类实现）"""
        pass

    @abstractmethod
    def process(self, frame, detections: List[Dict], timestamp: float) -> Optional[Dict]:
        """
        处理检测结果，返回报警信息

        Args:
            frame: 当前帧图像
            detections: 检测结果列表
                [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'conf': float,
                        'cls': int,
                        'track_id': int  # 可选
                    },
                    ...
                ]
            timestamp: 当前时间戳

        Returns:
            alarm_info: dict or None
                {
                    'rule_type': str,  # 规则类型
                    'timestamp': str,   # 报警时间
                    'frame': numpy array,  # 报警图像
                    'details': dict  # 规则特定信息
                }
        """
        pass

    @abstractmethod
    def reset(self):
        """重置规则状态（用于配置更新时）"""
        pass

    def should_alarm(self, timestamp: float) -> bool:
        """
        判断是否在冷却期

        Args:
            timestamp: 当前时间戳

        Returns:
            bool: 是否可以报警
        """
        if self.last_alarm_time is None:
            return True
        return (timestamp - self.last_alarm_time) >= self.repeated_alarm_time

    def trigger_alarm(self, alarm_info: Dict, api_client) -> bool:
        """
        触发报警（调用API）

        Args:
            alarm_info: 报警信息
            api_client: API客户端

        Returns:
            bool: 报警是否成功
        """
        try:
            # 更新报警时间
            self.last_alarm_time = time.time()

            # 调用API上传报警
            success = api_client.upload_alarm(alarm_info)

            return success

        except Exception as e:
            logger.error(f"报警上传异常: {e}", exc_info=True)
            return False

    def filter_by_confidence(self, detections: List[Dict]) -> List[Dict]:
        """
        根据sensitivity过滤检测结果

        Args:
            detections: 原始检测结果

        Returns:
            List[Dict]: 过滤后的检测结果
        """
        return [det for det in detections if det['conf'] >= self.sensitivity]

    def _create_alarm_data(self, alarm_type: str, alarm_type_name: str,
                          image_base64: str, additional_info: Dict = None) -> Dict:
        """
        创建标准报警数据格式

        Args:
            alarm_type: 报警类型代码
            alarm_type_name: 报警类型名称
            image_base64: Base64编码的图片
            additional_info: 额外信息

        Returns:
            Dict: 报警数据
        """
        alarm_data = {
            "deviceId": self.device_info.get('deviceId', ''),
            "deviceName": self.device_info.get('deviceName', ''),
            "deviceCode": self.device_info.get('deviceCode', ''),
            "deviceIp": self.device_info.get('deviceIp', ''),
            "channelId": self.device_info.get('channelId', ''),
            "channelName": self.device_info.get('channelName', ''),
            "channelCode": self.device_info.get('channelCode', ''),
            "alarmPicCode": image_base64,
            "nodeType": "2",
            "alarmDate": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "alarmType": alarm_type,
            "alarmTypeName": alarm_type_name
        }

        # 添加额外信息
        if additional_info:
            alarm_data.update(additional_info)

        return alarm_data
