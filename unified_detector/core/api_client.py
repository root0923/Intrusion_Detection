"""
API客户端 - 后端接口交互

功能：
- 登录认证
- 获取设备配置
- 获取视频流地址
- 上传报警信息
- 保活机制
"""
import requests
import threading
import time
import logging
from typing import Dict, Optional
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad


logger = logging.getLogger(__name__)


def aes_encrypt_password(password: str) -> str:
    """
    使用AES CBC模式加密密码（与后端Hutool AES加密保持一致）

    Args:
        password: 明文密码

    Returns:
        str: 十六进制编码的加密密文
    """
    # AES密钥和IV（固定值，与后端保持一致）
    AES_KEY = b'JzjPLY9632AijnEQ'  # 16字节
    AES_IV = b'DYgjCEIikmj2W9xN'   # 16字节

    try:
        # 创建AES加密器（CBC模式，PKCS7Padding）
        cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)

        # PKCS7填充
        padded_data = pad(password.encode('utf-8'), AES.block_size)

        # 加密
        encrypted_data = cipher.encrypt(padded_data)

        # 十六进制编码
        encrypted_hex = encrypted_data.hex()

        return encrypted_hex

    except Exception as e:
        logger.error(f"密码加密失败: {e}")
        raise


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
                logger.debug("✓ 获取设备配置成功")
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
