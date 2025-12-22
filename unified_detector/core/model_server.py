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
            device: 设备(cuda:0/cpu)
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

    def start(self):
        """启动模型服务器进程"""
        self.manager = Manager()
        self.request_queue = self.manager.Queue(maxsize=100)
        self.response_queues = self.manager.dict()

        self.server_process = Process(
            target=self._server_loop,
            args=(self.request_queue, self.response_queues,
                  self.model_yaml, self.model_weights,
                  self.thermal_model_yaml, self.thermal_model_weights,
                  self.device, self.tracker),
            daemon=True,
            name="ModelServer"
        )
        self.server_process.start()
        logger.info("✓ 模型服务器已启动（双模型：可见光+热成像）")

        # 等待模型加载完成
        time.sleep(8)  # 双模型需要更长的加载时间

    def stop(self):
        """停止模型服务器"""
        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            logger.info("✓ 模型服务器已停止")

    def register_client(self, client_id: str) -> Queue:
        """
        注册客户端

        Args:
            client_id: 客户端标识

        Returns:
            response_queue: 该客户端的响应队列
        """
        response_queue = self.manager.Queue()
        self.response_queues[client_id] = response_queue
        logger.info(f"✓ 客户端已注册: {client_id}")
        return response_queue

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
    def _server_loop(request_queue: Queue, response_queues: Dict,
                     model_yaml: str, model_weights: str,
                     thermal_model_yaml: str, thermal_model_weights: str,
                     device: str, tracker: str):
        """服务器主循环（在独立进程中运行）"""
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [ModelServer] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger(__name__)
        logger.info("模型服务器进程启动...")

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
        logger.info("开始监听推理请求...")

        while True:
            try:
                # 获取请求（阻塞，超时1秒）
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

                    # 发送响应
                    response = {
                        'client_id': client_id,
                        'detections': detections,
                        'timestamp': time.time()
                    }

                    response_queue = response_queues.get(client_id)
                    if response_queue:
                        try:
                            response_queue.put_nowait(response)
                            stats['successful'] += 1
                        except:
                            logger.warning(f"[{client_id}] 响应队列已满")
                            stats['failed'] += 1
                    else:
                        logger.warning(f"[{client_id}] 客户端未找到")
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
    """轻量级模型客户端（用于子进程）"""

    def __init__(self, client_id: str, request_queue, response_queues, model_type: str = 'visible'):
        """
        Args:
            client_id: 客户端标识（camera_key）
            request_queue: 共享的请求队列
            response_queues: 共享的响应队列字典(Manager.dict())
            model_type: 模型类型 ('visible'=可见光, 'thermal'=热成像)
        """
        self.client_id = client_id
        self.request_queue = request_queue
        self.response_queues = response_queues
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

        # 等待响应（从响应队列字典中获取）
        response_queue = self.response_queues.get(self.client_id)
        if not response_queue:
            logger.error(f"[{self.client_id}] 响应队列未找到")
            return []

        try:
            response = response_queue.get(timeout=timeout)
            return response['detections']
        except Empty:
            logger.warning(f"[{self.client_id}] 推理超时 ({timeout}s)")
            return []
