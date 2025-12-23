"""
模型推理服务器 - 集中式推理，进程间共享

功能：
- 单个进程加载模型
- 多个camera进程通过队列提交推理请求
- 避免重复加载模型，大幅节省内存

使用场景：
- 内存受限，需要运行多路视频流
- 多路共享同一个GPU模型
"""

import logging
import time
import numpy as np
from multiprocessing import Process, Queue, Manager
from queue import Empty
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelServer:
    """集中式模型推理服务器（支持双模型：可见光+热成像）"""

    def __init__(self, model_yaml: str, model_weights: str,
                 thermal_model_yaml: str, thermal_model_weights: str,
                 device: str, tracker: str = 'bytetrack'):
        """
        Args:
            model_yaml: 可见光模型配置文件
            model_weights: 可见光模型权重文件
            thermal_model_yaml: 热成像模型配置文件
            thermal_model_weights: 热成像模型权重文件
            device: 设备(cuda:0/cp)
            tracker: 跟踪器类型
        """
        self.model_yaml = model_yaml
        self.model_weights = model_weights
        self.thermal_model_yaml = thermal_model_yaml
        self.thermal_model_weights = thermal_model_weights
        self.device = device
        self.tracker = tracker

        # 队列
        self.request_queue = None  # 推理请求队列
        self.response_queues = {}  # {client_id: response_queue}
        self.manager = None
        self.server_process = None

        # 模型实例（在服务进程中加载）
        self.detector_visible = None  # 可见光模型
        self.detector_thermal = None  # 热成像模型

    def start(self, camera_keys=None, max_cameras=50):
        """
        启动模型服务器进程（Python 3.8兼容：预留槽位方案）

        Args:
            camera_keys: 需要预先分配的camera_key列表（可选）
            max_cameras: 最大支持的摄像头数量（预留槽位数）
        """
        self.manager = Manager()
        self.request_queue = self.manager.Queue(maxsize=100)

        # 注册队列：用于动态添加新摄像头（只传递camera_key和index，不传递Queue对象）
        self.registration_queue = self.manager.Queue()
        self.registration_ack_queue = self.manager.Queue()

        # 预留槽位方案：预先创建固定数量的响应队列（用索引0~max_cameras-1）
        logger.info(f"预留 {max_cameras} 个响应队列槽位（支持热更新）...")
        response_queues_by_index = {}
        for i in range(max_cameras):
            response_queues_by_index[i] = self.manager.Queue()

        # 映射表：camera_key → queue_index（主进程维护）
        self.camera_to_index = {}
        self.next_available_index = 0

        # 为已知camera分配槽位
        if camera_keys:
            logger.info(f"为 {len(camera_keys)} 个camera预先分配槽位...")
            for camera_key in camera_keys:
                if self.next_available_index < max_cameras:
                    self.camera_to_index[camera_key] = self.next_available_index
                    self.next_available_index += 1
                else:
                    logger.warning(f"槽位已满，无法分配: {camera_key}")
            logger.info(f"✓ 已分配 {len(camera_keys)} 个槽位")

        # 保存队列引用（主进程通过camera_key访问）
        self.response_queues_by_index = response_queues_by_index

        self.server_process = Process(
            target=self._server_loop,
            args=(self.request_queue, self.registration_queue, self.registration_ack_queue,
                  response_queues_by_index,
                  self.model_yaml, self.model_weights,
                  self.thermal_model_yaml, self.thermal_model_weights,
                  self.device, self.tracker, 0),
            daemon=True,
            name="ModelServer"
        )
        self.server_process.start()

        logger.info(f"✓ 模型服务器已启动（双模型，槽位: {len(camera_keys) if camera_keys else 0}/{max_cameras}）")

        # 等待模型加载完成
        time.sleep(8)  # 双模型需要更长的加载时间

    def stop(self):
        """停止模型服务器"""
        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            logger.info("✓ 模型服务器已停止")

    def register_client(self, client_id: str, timeout: float = 2.0) -> bool:
        """
        动态注册客户端（预留槽位方案，Python 3.8兼容）

        Args:
            client_id: 客户端标识
            timeout: 等待注册确认的超时时间（秒）

        Returns:
            是否注册成功
        """
        # 检查是否已注册
        if client_id in self.camera_to_index:
            logger.warning(f"客户端已注册: {client_id}")
            return True

        # 分配一个空闲槽位
        if self.next_available_index >= len(self.response_queues_by_index):
            logger.error(f"✗ 槽位已满，无法注册: {client_id}")
            return False

        queue_index = self.next_available_index
        self.next_available_index += 1

        # 通过注册队列发送注册请求（只传递camera_key和index，不传递Queue对象）
        self.registration_queue.put({
            'camera_key': client_id,
            'queue_index': queue_index
        })

        # 等待子进程确认注册完成
        try:
            ack = self.registration_ack_queue.get(timeout=timeout)
            if ack['camera_key'] == client_id and ack['status'] == 'success':
                # 保存映射到主进程
                self.camera_to_index[client_id] = queue_index
                logger.info(f"✓ 客户端已注册: {client_id} → 槽位{queue_index}")
                return True
            else:
                logger.error(f"✗ 客户端注册失败: {client_id}")
                return False
        except Empty:
            logger.error(f"✗ 客户端注册超时: {client_id}")
            return False

    def infer(self, client_id: str, frame: np.ndarray,
              conf_threshold: float = 0.25,
              iou_threshold: float = 0.7,
              target_size: int = 640,
              timeout: float = 5.0) -> Optional[list]:
        """
        推理请求（阻塞）

        Args:
            client_id: 客户端ID
            frame: 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            target_size: 推理尺寸
            timeout: 超时时间(秒)

        Returns:
            detections: 检测结果列表
        """
        # 构造请求
        request = {
            'client_id': client_id,
            'frame': frame,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'target_size': target_size,
            'timestamp': time.time()
        }

        # 提交请求
        try:
            self.request_queue.put(request, timeout=1.0)
        except:
            logger.error(f"[{client_id}] 请求队列已满，丢弃该帧")
            return None

        # 等待响应
        response_queue = self.response_queues.get(client_id)
        if not response_queue:
            logger.error(f"[{client_id}] 客户端未注册")
            return None

        try:
            response = response_queue.get(timeout=timeout)
            return response['detections']
        except Empty:
            logger.warning(f"[{client_id}] 推理超时 ({timeout}s)")
            return None

    @staticmethod
    def _server_loop(request_queue: Queue, registration_queue: Queue, registration_ack_queue: Queue,
                     response_queues_by_index: Dict,
                     model_yaml: str, model_weights: str,
                     thermal_model_yaml: str, thermal_model_weights: str,
                     device: str, tracker: str, gpu_idx: int = 0):
        """服务器主循环（预留槽位方案）"""
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [ModelServer] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger(__name__)
        logger.info(f"模型服务器进程启动（GPU{gpu_idx}: {device}）...")
        logger.info(f"已接收 {len(response_queues_by_index)} 个预留的响应队列槽位")

        # 维护camera_key → queue_index的映射表（子进程）
        camera_key_to_index = {}

        # 加载两个模型
        try:
            from unified_detector.core.detector import UnifiedDetector

            logger.info("正在加载可见光YOLO模型...")
            detector_visible = UnifiedDetector(model_yaml, model_weights, device, tracker)
            logger.info("✓ 可见光YOLO模型加载完成")

            logger.info("正在加载热成像YOLO模型...")
            detector_thermal = UnifiedDetector(thermal_model_yaml, thermal_model_weights, device, tracker)
            logger.info("✓ 热成像YOLO模型加载完成")

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {e}", exc_info=True)
            return

        # 统计信息
        stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'last_report_time': time.time()
        }

        # 主循环
        logger.info("开始监听推理请求和注册请求...")

        while True:
            try:
                # 1. 先处理注册队列（非阻塞）
                try:
                    while True:
                        reg_msg = registration_queue.get_nowait()
                        camera_key = reg_msg['camera_key']
                        queue_index = reg_msg['queue_index']

                        # 建立映射关系（不需要传递Queue对象）
                        camera_key_to_index[camera_key] = queue_index
                        logger.info(f"✓ 动态注册camera: {camera_key} → 槽位{queue_index}")

                        # 发送确认消息
                        registration_ack_queue.put({
                            'camera_key': camera_key,
                            'status': 'success'
                        })
                except Empty:
                    pass  # 注册队列为空，继续处理推理请求

                # 2. 再获取推理请求（阻塞，超时1秒）
                try:
                    request = request_queue.get(timeout=1.0)
                except Empty:
                    continue

                stats['total_requests'] += 1
                client_id = request['client_id']

                # 推理
                try:
                    frame = request['frame']
                    conf_threshold = request['conf_threshold']
                    iou_threshold = request['iou_threshold']
                    target_size = request['target_size']
                    model_type = request.get('model_type', 'visible')  # 'visible' 或 'thermal'

                    # 根据模型类型选择对应的检测器
                    if model_type == 'thermal':
                        detector = detector_thermal
                    else:
                        detector = detector_visible

                    # 执行检测
                    detections = detector.detect_and_track(
                        frame, conf_threshold, iou_threshold, target_size
                    )

                    # 发送响应（通过映射表查找队列）
                    response = {
                        'client_id': client_id,
                        'detections': detections,
                        'timestamp': time.time()
                    }

                    queue_index = camera_key_to_index.get(client_id)
                    if queue_index is not None:
                        response_queue = response_queues_by_index[queue_index]
                        try:
                            response_queue.put_nowait(response)
                            stats['successful'] += 1
                        except:
                            logger.warning(f"[{client_id}] 响应队列已满")
                            stats['failed'] += 1
                    else:
                        logger.warning(f"[{client_id}] 客户端未注册")
                        stats['failed'] += 1

                except Exception as e:
                    logger.error(f"[{client_id}] 推理异常: {e}")
                    stats['failed'] += 1

                # 每30秒打印统计
                if time.time() - stats['last_report_time'] > 30:
                    logger.info(f"统计: 总请求={stats['total_requests']}, "
                               f"成功={stats['successful']}, "
                               f"失败={stats['failed']}, "
                               f"队列长度={request_queue.qsize()}")
                    stats['last_report_time'] = time.time()

            except Exception as e:
                logger.error(f"服务器循环异常: {e}", exc_info=True)
                time.sleep(1)


class LightweightModelClient:
    """轻量级模型客户端（预留槽位方案）"""

    def __init__(self, client_id: str, request_queue, response_queues_by_index, camera_to_index, model_type: str = 'visible'):
        """
        Args:
            client_id: 客户端标识（camera_key）
            request_queue: 共享的请求队列
            response_queues_by_index: 按索引存储的响应队列字典
            camera_to_index: camera_key → queue_index 的映射表
            model_type: 模型类型 ('visible'=可见光, 'thermal'=热成像)
        """
        self.client_id = client_id
        self.request_queue = request_queue
        self.response_queues_by_index = response_queues_by_index
        self.camera_to_index = camera_to_index
        self.model_type = model_type

        logger.info(f"[{client_id}] 轻量级客户端已创建 (模型类型: {'热成像' if model_type == 'thermal' else '可见光'})")

    def detect_and_track(self, frame: np.ndarray,
                        conf_threshold: float = 0.25,
                        iou_threshold: float = 0.7,
                        target_size: int = 640,
                        timeout: float = 5.0) -> list:
        """
        检测和跟踪（接口兼容UnifiedDetector）

        Args:
            frame: 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            target_size: 推理尺寸
            timeout: 超时时间(秒)

        Returns:
            detections: 检测结果
        """
        # 构造请求
        request = {
            'client_id': self.client_id,
            'frame': frame,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'target_size': target_size,
            'model_type': self.model_type,  # 指定使用的模型类型
            'timestamp': time.time()
        }

        # 提交请求
        try:
            self.request_queue.put(request, timeout=1.0)
        except:
            logger.error(f"[{self.client_id}] 请求队列已满，丢弃该帧")
            return []

        # 等待响应（通过映射表查找响应队列）
        queue_index = self.camera_to_index.get(self.client_id)
        if queue_index is None:
            logger.error(f"[{self.client_id}] 响应队列未注册")
            return []

        response_queue = self.response_queues_by_index[queue_index]

        try:
            response = response_queue.get(timeout=timeout)
            return response['detections']
        except Empty:
            logger.warning(f"[{self.client_id}] 推理超时 ({timeout}s)")
            return []
