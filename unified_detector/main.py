"""
统一检测框架 - 主程序

功能：
- 多进程管理
- 配置轮询和热更新
- 进程生命周期管理
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Dict
import traceback

from core.api_client import APIClient
from core.processor import CameraProcessor
from core.model_server import ModelServer
from utils.config_parser import ConfigParser
import warnings
warnings.filterwarnings('ignore')


# ============ 配置日志 ============
def setup_logging(log_dir: Path):
    """配置日志"""
    log_dir.mkdir(exist_ok=True, parents=True)

    # 日志文件路径（按日期分割）
    log_file = log_dir / f"unified_detector_{datetime.now().strftime('%Y%m%d')}.log"

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # 控制台输出
            logging.StreamHandler(),
            # 文件输出
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    return logging.getLogger(__name__)


# ============ 进程管理器 ============
class ProcessManager:
    """多进程管理器（支持两种模式：ModelServer或多进程独立模型）"""

    def __init__(self, api_client: APIClient, model_yaml: str, model_weights: str,
                 thermal_model_yaml: str, thermal_model_weights: str,
                 devices: list, target_size: int, process_fps: float, tracker: str, logdir: Path,
                 use_model_server: bool = False,
                 enable_adaptive_fps: bool = False, fps_idle: float = 1.0,
                 fps_active: float = 5.0, person_timeout: int = 5,
                 tripwire_first_alarm_time: float = 10.0, tripwire_tolerance_time: float = 3.0):
        """
        Args:
            api_client: API客户端
            model_yaml: 可见光模型配置文件路径
            model_weights: 可见光模型权重文件路径
            thermal_model_yaml: 热成像模型配置文件路径
            thermal_model_weights: 热成像模型权重文件路径
            devices: 设备列表（例如 ['cuda:0', 'cuda:1']）
            target_size: 推理图像尺寸
            process_fps: 处理帧率
            tracker: 跟踪器类型
            logdir: 日志目录
            use_model_server: 是否使用集中式模型服务器
                - True: 节省显存（所有进程共享1个模型），但推理串行，延迟高
                - False: 每个进程独立加载模型，显存占用大，但推理并行，延迟低
            enable_adaptive_fps: 启用自适应帧率（仅对绊线入侵规则生效）
            fps_idle: 无人时的帧率
            fps_active: 有人时的帧率
            person_timeout: 多少秒没检测到人后切换到低帧率
            tripwire_first_alarm_time: 绊线入侵首次报警时间（秒）
            tripwire_tolerance_time: 绊线入侵容忍时间（秒）
        """
        self.api_client = api_client
        self.model_yaml = model_yaml
        self.model_weights = model_weights
        self.thermal_model_yaml = thermal_model_yaml
        self.thermal_model_weights = thermal_model_weights
        self.devices = devices if devices else ['cuda:0']
        self.target_size = target_size
        self.process_fps = process_fps
        self.tracker = tracker
        self.log_dir = logdir
        self.use_model_server = use_model_server

        # 动态帧率配置
        self.enable_adaptive_fps = enable_adaptive_fps
        self.fps_idle = fps_idle
        self.fps_active = fps_active
        self.person_timeout = person_timeout

        # 绊线入侵配置
        self.tripwire_first_alarm_time = tripwire_first_alarm_time
        self.tripwire_tolerance_time = tripwire_tolerance_time

        # 进程字典: {camera_key: {'process': Process, 'config_queue': Queue, 'config': Dict, 'device': str}}
        self.processes = {}

        # GPU分配计数器（用于轮询分配）
        self.device_counter = 0

        # 模型服务器（仅在use_model_server=True时使用）
        # 支持多GPU：每个GPU启动一个ModelServer
        self.model_servers = []  # 多个ModelServer实例（每个GPU一个）
        self.request_queues = []  # 每个ModelServer的请求队列
        self.response_queues_list = []  # 每个ModelServer的响应队列字典

        self.logger = logging.getLogger(__name__)

        # 日志输出GPU配置
        self.logger.info(f"使用 {len(self.devices)} 个GPU设备: {self.devices}")

        # 日志输出动态帧率配置
        if self.enable_adaptive_fps:
            self.logger.info(f"动态帧率已启用（仅对绊线入侵规则生效）")
            self.logger.info(f"  空闲帧率: {self.fps_idle}fps")
            self.logger.info(f"  活跃帧率: {self.fps_active}fps")
            self.logger.info(f"  人员超时: {self.person_timeout}秒")

        # 条件性启动模型服务器
        if self.use_model_server:
            self._start_model_server()
        else:
            self.logger.info("使用多进程独立模型模式（每个进程加载独立模型）")
            self.logger.info("  优点：推理并行，延迟低")
            self.logger.info("  缺点：GPU显存占用大（每路约1-2GB）")
            self.logger.info(f"  进程将在 {len(self.devices)} 个GPU上平均分配")

    def _start_model_server(self):
        """启动集中式模型服务器（支持多GPU）"""
        self.logger.info(f"正在启动集中式模型服务器（双模型）...")
        self.logger.info(f"  GPU数量: {len(self.devices)}")
        self.logger.info(f"  每个GPU将启动一个独立的ModelServer")
        self.logger.info(f"  进程将轮询分配到不同的ModelServer")
        self.logger.info(f"  这将大幅节省GPU显存（每路节省 ~1-2GB）")

        # 为每个GPU启动一个ModelServer
        for gpu_idx, device in enumerate(self.devices):
            self.logger.info(f"\n  启动GPU{gpu_idx}的ModelServer ({device})...")

            model_server = ModelServer(
                model_yaml=self.model_yaml,
                model_weights=self.model_weights,
                thermal_model_yaml=self.thermal_model_yaml,
                thermal_model_weights=self.thermal_model_weights,
                device=device,
                tracker=self.tracker
            )
            # 启动时不传camera_keys，后续在start_process时创建响应队列
            model_server.start()

            self.model_servers.append(model_server)
            self.request_queues.append(model_server.request_queue)
            self.response_queues_list.append(model_server.response_queues)

            self.logger.info(f"  ✓ GPU{gpu_idx} ModelServer已启动")

        self.logger.info(f"\n✓ 所有ModelServer已启动（{len(self.model_servers)} 个GPU，双模型：可见光+热成像）")
        self.logger.info(f"  预计总显存占用: {len(self.devices) * 1.0:.1f}GB（每GPU约1GB）")
        self.logger.info(f"  进程分配策略: 轮询到{len(self.devices)}个GPU")

    def _get_next_device(self):
        """获取下一个GPU设备（轮询分配）"""
        device = self.devices[self.device_counter % len(self.devices)]
        self.device_counter += 1
        return device

    def start_process(self, camera_config: Dict):
        """启动单个摄像头进程"""
        camera_key = camera_config['camera_key']

        if camera_key in self.processes:
            self.logger.warning(f"进程已存在: {camera_key}")
            return

        # 创建配置更新队列
        config_queue = Queue()

        # 根据userType选择对应的模型
        user_type = camera_config.get('user_type', '0')
        if user_type == '1':
            # 热成像通道
            selected_model_yaml = self.thermal_model_yaml
            selected_model_weights = self.thermal_model_weights
            model_type_str = "热成像模型"
        else:
            # 可见光通道（默认）
            selected_model_yaml = self.model_yaml
            selected_model_weights = self.model_weights
            model_type_str = "可见光模型"

        self.logger.info(f"[{camera_key}] 通道类型: {'热成像' if user_type == '1' else '可见光'} (userType={user_type})，使用{model_type_str}")

        # 根据模式准备参数
        if self.use_model_server:
            # ModelServer模式：轮询分配到不同的GPU的ModelServer
            gpu_idx = self.device_counter % len(self.devices)
            self.device_counter += 1

            request_queue = self.request_queues[gpu_idx]
            response_queues = self.response_queues_list[gpu_idx]
            assigned_device = self.devices[gpu_idx]

            # 预先创建响应队列
            if camera_key not in response_queues:
                response_queues[camera_key] = self.model_servers[gpu_idx].manager.Queue()

            self.logger.info(f"[{camera_key}] 分配到ModelServer: GPU{gpu_idx} ({assigned_device})")

            # 创建进程
            process = Process(
                target=camera_worker,
                args=(camera_config, selected_model_yaml, selected_model_weights,
                     assigned_device, self.target_size, self.process_fps,
                     self.api_client.base_url, self.api_client.token,
                     config_queue, self.tracker, self.log_dir,
                     True, request_queue, response_queues,
                     self.enable_adaptive_fps, self.fps_idle, self.fps_active, self.person_timeout,
                     self.tripwire_first_alarm_time, self.tripwire_tolerance_time),
                daemon=True,
                name=f"Camera-{camera_key}"
            )
        else:
            # 独立模型模式：每个进程加载独立模型
            assigned_device = self._get_next_device()

            self.logger.info(f"[{camera_key}] 分配GPU: {assigned_device}")
            # 独立模型模式：每个进程加载独立模型（使用分配的设备）
            process = Process(
                target=camera_worker,
                args=(camera_config, selected_model_yaml, selected_model_weights,
                     assigned_device, self.target_size, self.process_fps,
                     self.api_client.base_url, self.api_client.token,
                     config_queue, self.tracker, self.log_dir,
                     False, None, None,
                     self.enable_adaptive_fps, self.fps_idle, self.fps_active, self.person_timeout,
                     self.tripwire_first_alarm_time, self.tripwire_tolerance_time),
                daemon=True,
                name=f"Camera-{camera_key}"
            )

        process.start()

        self.processes[camera_key] = {
            'process': process,
            'config_queue': config_queue,
            'config': camera_config,
            'device': assigned_device
        }

        self.logger.info(f"✓ 启动进程: {camera_config['device_name']}/{camera_config['channel_name']} "
                        f"(规则: {list(camera_config['rules'].keys())})")

    def stop_process(self, camera_key: str):
        """停止单个摄像头进程"""
        if camera_key not in self.processes:
            self.logger.warning(f"进程不存在: {camera_key}")
            return

        proc_info = self.processes[camera_key]
        proc_info['process'].terminate()
        proc_info['process'].join(timeout=5)

        if proc_info['process'].is_alive():
            proc_info['process'].kill()
            self.logger.warning(f"强制终止进程: {camera_key}")

        del self.processes[camera_key]

        self.logger.info(f"✓ 停止进程: {camera_key}")

    def update_process(self, camera_key: str, new_camera_config: Dict):
        """更新进程配置（通过Queue热更新）"""
        if camera_key not in self.processes:
            self.logger.warning(f"进程不存在: {camera_key}")
            return

        proc_info = self.processes[camera_key]

        # 发送新配置到队列
        proc_info['config_queue'].put(new_camera_config)
        proc_info['config'] = new_camera_config

        self.logger.info(f"✓ 更新进程配置: {camera_key}")

    def stop_all(self):
        """停止所有进程"""
        self.logger.info("停止所有进程...")

        keys = list(self.processes.keys())
        for camera_key in keys:
            self.stop_process(camera_key)

        # 停止模型服务器（如果有）
        if self.model_servers:
            self.logger.info(f"正在停止 {len(self.model_servers)} 个ModelServer...")
            for idx, model_server in enumerate(self.model_servers):
                model_server.stop()
                self.logger.info(f"  ✓ GPU{idx} ModelServer已停止")
            self.logger.info("✓ 所有ModelServer已停止")

        self.logger.info("✓ 所有进程已停止")

    def get_status(self) -> Dict:
        """获取进程状态"""
        status = {
            'total': len(self.processes),
            'running': sum(1 for p in self.processes.values() if p['process'].is_alive()),
            'dead': sum(1 for p in self.processes.values() if not p['process'].is_alive())
        }
        return status


def camera_worker(camera_config: Dict, model_yaml: str, model_weights: str,
                 device: str, target_size: int, process_fps: float,
                 api_base_url: str, api_token: str,
                 config_queue: Queue, tracker: str, log_dir: Path,
                 use_model_server: bool = False,
                 request_queue=None, response_queues=None,
                 enable_adaptive_fps: bool = False, fps_idle: float = 1.0,
                 fps_active: float = 5.0, person_timeout: int = 5,
                 tripwire_first_alarm_time: float = 10.0, tripwire_tolerance_time: float = 3.0):
    """摄像头进程工作函数（支持两种模式）"""
    log_file = log_dir / f"unified_detector_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ],
        force=True
    )

    model_client = None

    if use_model_server:
        # ModelServer模式：创建轻量级客户端
        from core.model_server import LightweightModelClient

        # 根据user_type确定模型类型
        user_type = camera_config.get('user_type', '0')
        model_type = 'thermal' if user_type == '1' else 'visible'

        model_client = LightweightModelClient(
            camera_config['camera_key'],
            request_queue,
            response_queues,
            model_type=model_type
        )
        # 模型参数设为None
        model_yaml = None
        model_weights = None
        device = None
        tracker = None

    # 创建处理器
    processor = CameraProcessor(
        camera_config,
        model_yaml=model_yaml,
        model_weights=model_weights,
        device=device,
        target_size=target_size,
        process_fps=process_fps,
        api_base_url=api_base_url,
        api_token=api_token,
        config_queue=config_queue,
        tracker=tracker,
        model_client=model_client,
        enable_adaptive_fps=enable_adaptive_fps,
        fps_idle=fps_idle,
        fps_active=fps_active,
        person_timeout=person_timeout,
        tripwire_first_alarm_time=tripwire_first_alarm_time,
        tripwire_tolerance_time=tripwire_tolerance_time
    )
    processor.start()


# ============ 主程序 ============
def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='统一检测框架 - Unified Detection Framework')

    # API配置
    parser.add_argument('--api-url', type=str, default="http://10.16.7.79:9199",
                       help='后端API基础URL（如 http://localhost:8080）')
    parser.add_argument('--username', type=str, default='jzsx',
                       help='登录用户名')
    parser.add_argument('--password', type=str, default='JZSXKJ@2025',
                       help='登录密码')

    # 模型配置 - 可见光模型
    parser.add_argument('--model-yaml', type=str,
                       default="ultralytics/cfg/models/11/yolo11m.yaml",
                       help='可见光模型配置YAML文件')
    parser.add_argument('--weights', type=str,
                       default='data/LLVIP-yolo11m-e300-16-pretrained.pt',
                       help='可见光模型权重文件')

    # 模型配置 - 热成像模型
    parser.add_argument('--thermal-model-yaml', type=str,
                       default="ultralytics/cfg/models/11/yolo11x.yaml",
                       help='热成像模型配置YAML文件')
    parser.add_argument('--thermal-weights', type=str,
                       default='data/LLVIP_IF-yolo11x-e300-16-pretrained.pt',
                       help='热成像模型权重文件')

    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0', 'cuda:1'],
                       help='GPU设备列表 (例如: cuda:0 cuda:1，支持多GPU平均分配进程)')

    # 检测配置
    parser.add_argument('--target-size', type=int, default=640,
                       help='YOLO检测目标尺寸')
    parser.add_argument('--process-fps', type=float, default=1.0,
                       help='每秒处理帧数（抽帧）')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'botsort'],
                       help='跟踪器类型')

    # 性能优化
    parser.add_argument('--use-model-server', action='store_true', default=False,
                       help='使用集中式模型服务器（节省GPU显存，但推理为串行，延迟较高）')

    # 动态帧率（仅对绊线入侵规则生效）
    parser.add_argument('--enable-adaptive-fps', action='store_true', default=False,
                       help='启用自适应帧率（仅对启用绊线入侵检测的进程生效）')
    parser.add_argument('--fps-idle', type=float, default=1.0,
                       help='无人时的帧率（默认1.0fps）')
    parser.add_argument('--fps-active', type=float, default=5.0,
                       help='有人时的帧率（默认5.0fps）')
    parser.add_argument('--person-timeout', type=int, default=5,
                       help='多少秒没检测到人后切换到低帧率（默认5秒）')

    # 绊线入侵首次报警时间（避免启动时误报）
    parser.add_argument('--tripwire-first-alarm-time', type=float, default=5.0,
                       help='绊线入侵首次报警时间（秒），目标持续在危险侧多久后才报警，避免启动时误报（默认10秒）')
    parser.add_argument('--tripwire-tolerance-time', type=float, default=3.0,
                       help='绊线入侵容忍时间（秒），目标短暂消失后多久重置状态（默认3秒）')

    # 配置更新
    parser.add_argument('--config-update-interval', type=int, default=30,
                       help='配置更新间隔（秒）')

    # 日志
    parser.add_argument('--log-dir', type=str, default=None,
                       help='日志目录（默认: unified_detector/log）')

    args = parser.parse_args()

    # 配置日志
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).parent / 'log'
    logger = setup_logging(log_dir)

    logger.info("="*60)
    logger.info("统一检测框架 - Unified Detection Framework")
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

    current_configs = ConfigParser.parse_device_config(config_data)
    logger.info(f"✓ 解析配置成功: {len(current_configs)} 个摄像头")

    # 4. 启动进程管理器
    logger.info("\n[4/5] 启动进程管理器...")
    process_manager = ProcessManager(
        api_client=api_client,
        model_yaml=args.model_yaml,
        model_weights=args.weights,
        thermal_model_yaml=args.thermal_model_yaml,
        thermal_model_weights=args.thermal_weights,
        devices=args.devices,
        target_size=args.target_size,
        process_fps=args.process_fps,
        tracker=args.tracker,
        logdir=log_dir,
        use_model_server=args.use_model_server,
        enable_adaptive_fps=args.enable_adaptive_fps,
        fps_idle=args.fps_idle,
        fps_active=args.fps_active,
        person_timeout=args.person_timeout,
        tripwire_first_alarm_time=args.tripwire_first_alarm_time,
        tripwire_tolerance_time=args.tripwire_tolerance_time
    )

    # 启动所有摄像头进程
    for camera_key, camera_config in current_configs.items():
        process_manager.start_process(camera_config)

    status = process_manager.get_status()
    logger.info(f"✓ 进程状态: {status}")

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

            new_configs = ConfigParser.parse_device_config(new_config_data)

            # 比对配置变化
            changes = ConfigParser.compare_configs(current_configs, new_configs)

            # 处理新增的摄像头
            if changes['add']:
                logger.info(f"新增摄像头: {len(changes['add'])}")
                for camera_key in changes['add']:
                    process_manager.start_process(new_configs[camera_key])

            # 处理删除的摄像头
            if changes['remove']:
                logger.info(f"删除摄像头: {len(changes['remove'])}")
                for camera_key in changes['remove']:
                    process_manager.stop_process(camera_key)

            # 处理更新的摄像头
            if changes['update']:
                logger.info(f"更新摄像头: {len(changes['update'])}")
                for camera_key in changes['update']:
                    process_manager.update_process(camera_key, new_configs[camera_key])

            if not any(changes.values()):
                logger.debug("配置无变化")

            # 更新当前配置
            current_configs = new_configs

            # 打印状态
            status = process_manager.get_status()
            logger.info(f"当前状态: {status}")

    except KeyboardInterrupt:
        logger.info("\n\n用户中断，正在退出...")

    except Exception as e:
        logger.error(f"主循环异常: {e}")
        traceback.print_exc()

    finally:
        # 清理
        logger.info("\n清理资源...")
        process_manager.stop_all()
        api_client.stop_keep_alive()
        logger.info("✓ 程序退出")


if __name__ == '__main__':
    main()