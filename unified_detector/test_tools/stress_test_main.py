"""
多路并发压测脚本（支持两种模式）

功能：
- 从API获取真实配置（1路）
- 复制N份模拟N路并发
- 所有路都指向同一个RTSP流
- 监控GPU显存和处理延迟

模式1：ModelServer模式（--use-model-server）
  - 所有进程共享1-2个模型
  - 节省显存，但推理串行

模式2：独立模型模式（默认）
  - 每个进程加载独立模型
  - 支持多GPU自动分配（--gpu-devices 0,1）
  - 显存占用大，但推理完全并行
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
from unified_detector.core.model_server import ModelServer
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
                 thermal_model_yaml, thermal_model_weights,
                 device, target_size, process_fps,
                 api_base_url, api_token, config_queue, tracker, log_dir,
                 use_model_server=False, request_queue=None, response_queues=None):
    """摄像头进程工作函数（支持两种模式）"""
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

    model_client = None

    if use_model_server:
        # ModelServer模式：创建轻量级客户端
        from unified_detector.core.model_server import LightweightModelClient

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
    else:
        # 独立模型模式：根据user_type选择对应模型
        user_type = camera_config.get('user_type', '0')
        if user_type == '1':
            # 热成像通道
            model_yaml = thermal_model_yaml
            model_weights = thermal_model_weights

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
        model_client=model_client
    )
    processor.start()


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='20路并发压测工具')

    # API配置
    parser.add_argument('--api-url', type=str, default="http://10.16.7.79:9199",
                       help='后端API基础URL')
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
                       default="ultralytics/cfg/models/11/yolo11n.yaml",
                       help='热成像模型配置YAML文件')
    parser.add_argument('--thermal-weights', type=str,
                       default='data/LLVIP-yolo11n-e300-16-pretrained-.pt',
                       help='热成像模型权重文件')

    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0', 'cuda:1'],
                       help='GPU设备列表 (例如: cuda:0 cuda:1，进程将轮询分配到各GPU)')

    # 检测配置
    parser.add_argument('--target-size', type=int, default=640,
                       help='YOLO检测目标尺寸')
    parser.add_argument('--process-fps', type=float, default=5,
                       help='每秒处理帧数')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       help='跟踪器类型')

    # 性能优化
    parser.add_argument('--use-model-server', action='store_true', default=True,
                       help='使用集中式模型服务器（节省显存，支持多GPU，每GPU启动一个ModelServer）')
    parser.add_argument('--no-model-server', dest='use_model_server', action='store_false',
                       help='使用独立模型模式（每进程独立模型，显存占用大但延迟低）')

    # 压测配置
    parser.add_argument('--num-streams', type=int, default=24,
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
    logger.info(f"多路并发压测工具 - {args.num_streams} 路并发")
    logger.info("="*60)
    logger.info(f"模式: {'ModelServer（共享模型）' if args.use_model_server else '独立模型（多进程）'}")
    logger.info(f"GPU设备: {args.devices}")
    logger.info(f"处理帧率: {args.process_fps} fps")

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

    # 显示所有可用的摄像头
    logger.info(f"✓ 获取到 {len(real_configs)} 个真实摄像头配置:")
    for idx, (key, config) in enumerate(real_configs.items()):
        logger.info(f"  [{idx+1}] {config['device_name']}/{config['channel_name']} - 规则: {list(config['rules'].keys())}")

    # 3. 复制配置生成N路
    logger.info(f"\n[3/4] 生成 {args.num_streams} 路配置...")

    # 计算每个真实摄像头需要复制多少份
    num_real_cameras = len(real_configs)
    copies_per_camera = args.num_streams // num_real_cameras
    extra_copies = args.num_streams % num_real_cameras

    logger.info(f"  真实摄像头数量: {num_real_cameras}")
    logger.info(f"  每个摄像头复制: {copies_per_camera} 份")
    if extra_copies > 0:
        logger.info(f"  前 {extra_copies} 个摄像头额外复制1份")

    test_configs = {}
    template_configs = list(real_configs.values())

    for cam_idx, template_config in enumerate(template_configs):
        # 确定这个摄像头需要复制多少份
        num_copies = copies_per_camera + (1 if cam_idx < extra_copies else 0)

        logger.info(f"\n  摄像头 {cam_idx+1}: {template_config['device_name']}/{template_config['channel_name']}")
        logger.info(f"    复制 {num_copies} 份")

        for copy_idx in range(num_copies):
            # 深拷贝配置
            config = copy.deepcopy(template_config)

            # 修改camera_key使其唯一
            original_key = template_config['camera_key']
            config['camera_key'] = f"{original_key}_copy{copy_idx:02d}"
            config['device_name'] = f"{template_config['device_name']}_副本{copy_idx:02d}"
            config['channel_name'] = f"{template_config['channel_name']}_副本{copy_idx:02d}"

            test_configs[config['camera_key']] = config

    logger.info(f"\n✓ 生成 {len(test_configs)} 路配置（{num_real_cameras} 个真实流，每个复制 {copies_per_camera}~{copies_per_camera+1} 份）")

    # 3.5. 条件性启动模型服务器
    model_servers = []  # 多个ModelServer实例（每个GPU一个）
    request_queues = []  # 每个ModelServer的请求队列
    response_queues_list = []  # 每个ModelServer的响应队列字典

    if args.use_model_server:
        logger.info(f"\n[3.5/5] 启动集中式模型服务器...")
        logger.info(f"  GPU数量: {len(args.devices)}")
        logger.info(f"  每个GPU将启动一个独立的ModelServer")
        logger.info(f"  进程将轮询分配到不同的ModelServer")

        # 先规划每个GPU需要处理哪些camera（轮询分配）
        camera_assignments = {}  # {gpu_idx: [camera_keys]}
        for gpu_idx in range(len(args.devices)):
            camera_assignments[gpu_idx] = []

        for idx, camera_key in enumerate(test_configs.keys()):
            gpu_idx = idx % len(args.devices)
            camera_assignments[gpu_idx].append(camera_key)

        # 为每个GPU启动一个ModelServer
        for gpu_idx, device in enumerate(args.devices):
            logger.info(f"\n  启动GPU{gpu_idx}的ModelServer ({device})...")

            # 获取分配给这个GPU的camera列表
            assigned_cameras = camera_assignments[gpu_idx]

            model_server = ModelServer(
                model_yaml=args.model_yaml,
                model_weights=args.weights,
                thermal_model_yaml=args.thermal_model_yaml,
                thermal_model_weights=args.thermal_weights,
                device=device,
                tracker=args.tracker
            )
            # 传递camera_keys，让ModelServer在启动前创建响应队列
            model_server.start(camera_keys=assigned_cameras)

            model_servers.append(model_server)
            request_queues.append(model_server.request_queue)
            response_queues_list.append(model_server.response_queues)

            logger.info(f"  ✓ GPU{gpu_idx} ModelServer已启动（{len(assigned_cameras)} 个camera）")

        logger.info(f"\n✓ 所有ModelServer已启动（{len(model_servers)} 个GPU，双模型：可见光+热成像）")
        logger.info(f"  预计总显存占用: {len(args.devices) * 1.0:.1f}GB（每GPU约1GB）")
        logger.info(f"  进程分配策略: 轮询到{len(args.devices)}个GPU")
    else:
        logger.info(f"\n[3.5/5] 使用独立模型模式（每进程加载独立模型）")
        logger.info(f"  GPU数量: {len(args.devices)}")
        logger.info(f"  GPU设备: {args.devices}")
        logger.info(f"  预计显存占用: 每GPU约 {(args.num_streams / len(args.devices)) * 0.53:.1f}GB")
        logger.info(f"  优点：推理并行，延迟低")
        logger.info(f"  缺点：GPU显存占用大（每路约1-2GB）")
        logger.info(f"  进程将在 {len(args.devices)} 个GPU上轮询分配")

    # 4. 启动所有进程
    logger.info(f"\n[4/5] 启动 {args.num_streams} 个进程...")
    processes = {}

    for idx, (camera_key, camera_config) in enumerate(test_configs.items()):
        config_queue = Queue()

        # 根据模式准备参数
        if args.use_model_server:
            # ModelServer模式：轮询分配到不同的GPU的ModelServer
            gpu_idx = idx % len(args.devices)
            request_queue = request_queues[gpu_idx]
            response_queues = response_queues_list[gpu_idx]
            assigned_gpu = args.devices[gpu_idx]

            process = Process(
                target=camera_worker,
                args=(camera_config, args.model_yaml, args.weights,
                     args.thermal_model_yaml, args.thermal_weights,
                     None, args.target_size, args.process_fps,
                     args.api_url, api_client.token,
                     config_queue, args.tracker, log_dir,
                     True, request_queue, response_queues),
                daemon=True,
                name=f"Camera-{camera_key}"
            )

            logger.info(f"✓ 启动进程 {idx+1}/{args.num_streams}: {camera_key} → GPU{gpu_idx} ({assigned_gpu})")
        else:
            # 独立模型模式：轮询分配GPU
            device = args.devices[idx % len(args.devices)]

            process = Process(
                target=camera_worker,
                args=(camera_config, args.model_yaml, args.weights,
                     args.thermal_model_yaml, args.thermal_weights,
                     device, args.target_size, args.process_fps,
                     args.api_url, api_client.token,
                     config_queue, args.tracker, log_dir,
                     False, None, None),
                daemon=True,
                name=f"Camera-{camera_key}"
            )

            logger.info(f"✓ 启动进程 {idx+1}/{args.num_streams}: {camera_key} → {device}")

        process.start()
        processes[camera_key] = process

        # 独立模型模式需要错峰加载，避免显存OOM
        if not args.use_model_server:
            time.sleep(1.5)  # 每个进程加载模型需要时间
        else:
            time.sleep(0.2)  # ModelServer模式无需加载模型，快速启动

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

        # 停止所有camera进程
        for camera_key, process in processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    logger.warning(f"强制终止进程: {camera_key}")

        logger.info("✓ 所有camera进程已停止")

        # 停止模型服务器（如果有）
        if args.use_model_server and model_servers:
            logger.info(f"正在停止 {len(model_servers)} 个ModelServer...")
            for idx, model_server in enumerate(model_servers):
                model_server.stop()
                logger.info(f"  ✓ GPU{idx} ModelServer已停止")
            logger.info("✓ 所有ModelServer已停止")


if __name__ == '__main__':
    main()
