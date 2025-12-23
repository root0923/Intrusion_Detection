"""
GPU性能监控脚本

功能：
- 实时监控GPU利用率、显存占用
- 监控系统CPU、内存占用
- 定期输出统计信息
- 保存到CSV文件
"""
import time
import psutil
import subprocess
import csv
from datetime import datetime
from pathlib import Path
import argparse


def get_gpu_info(gpu_ids=None):
    """
    使用nvidia-smi获取GPU信息

    Args:
        gpu_ids: GPU ID列表，如 [0, 1]。如果为None则查询所有GPU

    Returns:
        list: GPU信息列表，每个元素是一个dict（利用率、显存等）
    """
    try:
        # 构建nvidia-smi命令
        cmd = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
               '--format=csv,noheader,nounits']

        if gpu_ids is not None:
            # 指定GPU ID
            cmd.insert(1, f"--id={','.join(map(str, gpu_ids))}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            gpu_infos = []

            for line in output_lines:
                if not line.strip():
                    continue

                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id, gpu_util, mem_used, mem_total, temp = parts[:5]

                    gpu_infos.append({
                        'gpu_id': int(gpu_id),
                        'gpu_utilization': float(gpu_util),
                        'memory_used_mb': float(mem_used),
                        'memory_total_mb': float(mem_total),
                        'memory_percent': (float(mem_used) / float(mem_total)) * 100,
                        'temperature': float(temp)
                    })

            return gpu_infos if gpu_infos else None
        else:
            return None

    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return None


def get_system_info():
    """
    获取系统信息（CPU、内存）

    Returns:
        dict: 系统信息
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        return {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent
        }

    except Exception as e:
        print(f"获取系统信息失败: {e}")
        return None


def monitor_loop(interval: int, duration: int, output_file: Path, gpu_ids=None):
    """
    监控循环

    Args:
        interval: 监控间隔（秒）
        duration: 监控时长（秒），0表示无限
        output_file: 输出CSV文件路径
        gpu_ids: GPU ID列表，如 [0, 1]。如果为None则监控所有GPU
    """
    print("="*60)
    print("GPU性能监控工具")
    print("="*60)

    # 先获取一次GPU信息，确定GPU数量和ID
    test_gpu_infos = get_gpu_info(gpu_ids)
    if not test_gpu_infos:
        print("错误：无法获取GPU信息，请检查nvidia-smi是否可用")
        return

    num_gpus = len(test_gpu_infos)
    actual_gpu_ids = [info['gpu_id'] for info in test_gpu_infos]

    print(f"将监控 {num_gpus} 个GPU: {actual_gpu_ids}")

    # 创建CSV文件，动态生成列头
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # 构建列头
        headers = ['timestamp']
        for gpu_id in actual_gpu_ids:
            headers.extend([
                f'gpu{gpu_id}_util_%',
                f'gpu{gpu_id}_mem_used_mb',
                f'gpu{gpu_id}_mem_total_mb',
                f'gpu{gpu_id}_mem_%',
                f'gpu{gpu_id}_temp_c'
            ])
        headers.extend(['cpu_%', 'sys_mem_used_gb', 'sys_mem_total_gb', 'sys_mem_%'])

        writer.writerow(headers)

    print(f"监控数据将保存到: {output_file}")
    print(f"监控间隔: {interval}秒")
    print(f"监控时长: {'无限' if duration == 0 else f'{duration}秒'}")
    print("按 Ctrl+C 退出\n")

    start_time = time.time()

    try:
        while True:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            elapsed = time.time() - start_time

            # 获取GPU信息
            gpu_infos = get_gpu_info(gpu_ids)

            # 获取系统信息
            sys_info = get_system_info()

            if gpu_infos and sys_info:
                # 打印到控制台
                console_output = f"[{current_time}] "

                for gpu_info in gpu_infos:
                    console_output += (
                        f"GPU{gpu_info['gpu_id']}: {gpu_info['gpu_utilization']:.0f}% | "
                        f"显存: {gpu_info['memory_used_mb']:.0f}/{gpu_info['memory_total_mb']:.0f}MB "
                        f"({gpu_info['memory_percent']:.1f}%) | "
                        f"温度: {gpu_info['temperature']:.0f}°C | "
                    )

                console_output += (
                    f"CPU: {sys_info['cpu_percent']:.0f}% | "
                    f"内存: {sys_info['memory_used_gb']:.1f}/{sys_info['memory_total_gb']:.1f}GB "
                    f"({sys_info['memory_percent']:.1f}%)"
                )

                print(console_output)

                # 写入CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [current_time]

                    for gpu_info in gpu_infos:
                        row.extend([
                            gpu_info['gpu_utilization'],
                            gpu_info['memory_used_mb'],
                            gpu_info['memory_total_mb'],
                            gpu_info['memory_percent'],
                            gpu_info['temperature']
                        ])

                    row.extend([
                        sys_info['cpu_percent'],
                        sys_info['memory_used_gb'],
                        sys_info['memory_total_gb'],
                        sys_info['memory_percent']
                    ])

                    writer.writerow(row)

            # 检查是否达到时长
            if duration > 0 and elapsed >= duration:
                print(f"\n✓ 监控完成（{elapsed:.0f}秒）")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n用户中断，退出监控")

    print(f"\n监控数据已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='GPU性能监控工具')

    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='要监控的GPU ID列表，逗号分隔（如"0,1"），默认监控所有GPU')
    parser.add_argument('--interval', type=int, default=2,
                       help='监控间隔（秒）')
    parser.add_argument('--duration', type=int, default=0,
                       help='监控时长（秒），0表示无限')
    parser.add_argument('--output', type=str, default=None,
                       help='输出CSV文件路径')

    args = parser.parse_args()

    # 解析GPU ID列表
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    # 默认输出文件
    if args.output:
        output_file = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'logs'
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"gpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    monitor_loop(args.interval, args.duration, output_file, gpu_ids)


if __name__ == '__main__':
    main()