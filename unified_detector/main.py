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
from utils.config_parser import ConfigParser


# ============ 配置日志 ============
def setup_logging(log_dir: str = None):
    """配置日志"""
    if log_dir is None:
        log_dir = Path(__file__).parent / 'log'
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(exist_ok=True, parents=True)

    # 日志文件路径（按日期分割）
    log_file = log_dir / f"unified_detector_{datetime.now().strftime('%Y%m%d')}.log"

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

    return logging.getLogger(__name__)


# ============ 进程管理器 ============
class ProcessManager:
    """多进程管理器"""

    def __init__(self, api_client: APIClient, model_yaml: str, model_weights: str,
                 device: str, target_size: int, process_fps: float, tracker: str):
        """
        Args:
            api_client: API客户端
            model_yaml: 模型配置文件路径
            model_weights: 模型权重文件路径
            device: 设备
            target_size: 推理图像尺寸
            process_fps: 处理帧率
            tracker: 跟踪器类型
        """
        self.api_client = api_client
        self.model_yaml = model_yaml
        self.model_weights = model_weights
        self.device = device
        self.target_size = target_size
        self.process_fps = process_fps
        self.tracker = tracker

        # 进程字典: {camera_key: {'process': Process, 'config_queue': Queue, 'config': Dict}}
        self.processes = {}

        self.logger = logging.getLogger(__name__)

    def start_process(self, camera_config: Dict):
        """启动单个摄像头进程"""
        camera_key = camera_config['camera_key']

        if camera_key in self.processes:
            self.logger.warning(f"进程已存在: {camera_key}")
            return

        # 创建配置更新队列
        config_queue = Queue()

        # 创建进程
        process = Process(
            target=camera_worker,
            args=(camera_config, self.model_yaml, self.model_weights,
                 self.device, self.target_size, self.process_fps,
                 self.api_client.base_url, self.api_client.token,
                 config_queue, self.tracker),
            daemon=True,
            name=f"Camera-{camera_key}"
        )

        process.start()

        self.processes[camera_key] = {
            'process': process,
            'config_queue': config_queue,
            'config': camera_config
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
                 config_queue: Queue, tracker: str):
    """摄像头进程工作函数"""
    processor = CameraProcessor(
        camera_config, model_yaml, model_weights,
        device, target_size, process_fps,
        api_base_url, api_token,
        config_queue, tracker
    )
    processor.start()


# ============ 主程序 ============
def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='统一检测框架 - Unified Detection Framework')

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
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'botsort'],
                       help='跟踪器类型')

    # 配置更新
    parser.add_argument('--config-update-interval', type=int, default=30,
                       help='配置更新间隔（秒）')

    # 日志
    parser.add_argument('--log-dir', type=str, default=None,
                       help='日志目录（默认: unified_detector/log）')

    args = parser.parse_args()

    # 配置日志
    logger = setup_logging(args.log_dir)

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
        device=args.device,
        target_size=args.target_size,
        process_fps=args.process_fps,
        tracker=args.tracker
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
