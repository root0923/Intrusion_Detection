"""
20路并发压测脚本

功能：
- 从API获取真实配置（1路）
- 复制20份模拟20路并发
- 所有路都指向同一个RTSP流
- 监控GPU显存和处理延迟
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import logging
import copy
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue

from unified_detector.core.api_client import APIClient
from unified_detector.core.processor import CameraProcessor
from unified_detector.utils.config_parser import ConfigParser
import warnings
warnings.filterwarnings('ignore')


def setup_logging(log_dir: Path):
    """配置日志"""
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    return logging.getLogger(__name__)


def camera_worker(camera_config, model_yaml, model_weights,
                 device, target_size, process_fps,
                 api_base_url, api_token,
                 config_queue, tracker, log_dir):
    """摄像头进程工作函数"""
    log_file = log_dir / f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ],
        force=True
    )

    processor = CameraProcessor(
        camera_config, model_yaml, model_weights,
        device, target_size, process_fps,
        api_base_url, api_token,
        config_queue, tracker
    )
    processor.start()


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='20路并发压测工具')

    # API配置
    parser.add_argument('--api-url', type=str, default="http://localhost:9199",
                       help='后端API基础URL')
    parser.add_argument('--username', type=str, default='jzsx',
                       help='登录用户名')
    parser.add_argument('--password', type=str, default='JZSXKJ@2025',
                       help='登录密码')

    # 模型配置
    parser.add_argument('--model-yaml', type=str,
                       default="ultralytics/cfg/models/11/yolo11m.yaml",
                       help='模型配置YAML文件')
    parser.add_argument('--weights', type=str,
                       default='data/LLVIP-yolo11m-e300-16-pretrained.pt',
                       help='模型权重文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0 或 cpu)')

    # 检测配置
    parser.add_argument('--target-size', type=int, default=640,
                       help='YOLO检测目标尺寸')
    parser.add_argument('--process-fps', type=float, default=10.0,
                       help='每秒处理帧数')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       help='跟踪器类型')

    # 压测配置
    parser.add_argument('--num-streams', type=int, default=20,
                       help='并发流数量')
    parser.add_argument('--duration', type=int, default=300,
                       help='测试时长（秒）')

    # 日志
    parser.add_argument('--log-dir', type=str, default=None,
                       help='日志目录')

    args = parser.parse_args()

    # 配置日志
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).parent / 'logs'
    logger = setup_logging(log_dir)

    logger.info("="*60)
    logger.info(f"20路并发压测工具 - {args.num_streams} 路并发")
    logger.info("="*60)

    # 1. 登录
    logger.info("\n[1/4] 登录后端系统...")
    api_client = APIClient(args.api_url, args.username, args.password)

    if not api_client.login():
        logger.error("登录失败，程序退出")
        return

    # 2. 获取真实配置（应该只有1个摄像头）
    logger.info("\n[2/4] 获取真实配置...")
    config_data = api_client.get_device_config()

    if not config_data:
        logger.error("获取配置失败，程序退出")
        return

    real_configs = ConfigParser.parse_device_config(config_data)

    if len(real_configs) == 0:
        logger.error("没有找到启用的摄像头配置，程序退出")
        return

    # 获取第一个摄像头配置作为模板
    template_config = list(real_configs.values())[0]
    logger.info(f"✓ 使用模板配置: {template_config['device_name']}/{template_config['channel_name']}")
    logger.info(f"  规则: {list(template_config['rules'].keys())}")

    # 3. 复制配置生成N路
    logger.info(f"\n[3/4] 生成 {args.num_streams} 路配置...")
    test_configs = {}

    for i in range(args.num_streams):
        # 深拷贝配置
        config = copy.deepcopy(template_config)

        # 修改camera_key使其唯一
        config['camera_key'] = f"{template_config['camera_key']}_test_{i:02d}"
        config['device_name'] = f"{template_config['device_name']}_测试{i:02d}"
        config['channel_name'] = f"{template_config['channel_name']}_测试{i:02d}"

        test_configs[config['camera_key']] = config

    logger.info(f"✓ 生成 {len(test_configs)} 路配置")

    # 4. 启动所有进程
    logger.info(f"\n[4/4] 启动 {args.num_streams} 个进程...")
    processes = {}

    for camera_key, camera_config in test_configs.items():
        config_queue = Queue()

        process = Process(
            target=camera_worker,
            args=(camera_config, args.model_yaml, args.weights,
                 args.device, args.target_size, args.process_fps,
                 args.api_url, api_client.token,
                 config_queue, args.tracker, log_dir),
            daemon=True,
            name=f"Camera-{camera_key}"
        )

        process.start()
        processes[camera_key] = process

        logger.info(f"✓ 启动进程 {camera_key}")

    logger.info(f"\n✓ 所有进程已启动，共 {len(processes)} 个")
    logger.info(f"测试将运行 {args.duration} 秒...")
    logger.info("按 Ctrl+C 提前退出\n")

    # 5. 等待测试完成
    try:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time

            # 检查进程存活状态
            alive_count = sum(1 for p in processes.values() if p.is_alive())
            dead_count = len(processes) - alive_count

            logger.info(f"运行时间: {elapsed:.0f}s / {args.duration}s | "
                       f"进程状态: {alive_count} 存活, {dead_count} 异常")

            # 达到测试时长
            if elapsed >= args.duration:
                logger.info("\n✓ 测试完成")
                break

            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("\n\n用户中断，正在退出...")

    finally:
        # 6. 清理所有进程
        logger.info("\n清理资源...")
        for camera_key, process in processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    logger.warning(f"强制终止进程: {camera_key}")

        logger.info("✓ 所有进程已停止")


if __name__ == '__main__':
    main()
