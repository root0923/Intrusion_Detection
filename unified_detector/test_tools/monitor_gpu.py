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


def get_gpu_info():
    """
    使用nvidia-smi获取GPU信息

    Returns:
        dict: GPU信息（利用率、显存等）
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            gpu_util, mem_used, mem_total, temp = output.split(', ')

            return {
                'gpu_utilization': float(gpu_util),
                'memory_used_mb': float(mem_used),
                'memory_total_mb': float(mem_total),
                'memory_percent': (float(mem_used) / float(mem_total)) * 100,
                'temperature': float(temp)
            }
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


def monitor_loop(interval: int, duration: int, output_file: Path):
    """
    监控循环

    Args:
        interval: 监控间隔（秒）
        duration: 监控时长（秒），0表示无限
        output_file: 输出CSV文件路径
    """
    print("="*60)
    print("GPU性能监控工具")
    print("="*60)

    # 创建CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'gpu_util_%',
            'gpu_mem_used_mb',
            'gpu_mem_total_mb',
            'gpu_mem_%',
            'gpu_temp_c',
            'cpu_%',
            'sys_mem_used_gb',
            'sys_mem_total_gb',
            'sys_mem_%'
        ])

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
            gpu_info = get_gpu_info()

            # 获取系统信息
            sys_info = get_system_info()

            if gpu_info and sys_info:
                # 打印到控制台
                print(f"[{current_time}] "
                      f"GPU: {gpu_info['gpu_utilization']:.0f}% | "
                      f"显存: {gpu_info['memory_used_mb']:.0f}/{gpu_info['memory_total_mb']:.0f}MB "
                      f"({gpu_info['memory_percent']:.1f}%) | "
                      f"温度: {gpu_info['temperature']:.0f}°C | "
                      f"CPU: {sys_info['cpu_percent']:.0f}% | "
                      f"内存: {sys_info['memory_used_gb']:.1f}/{sys_info['memory_total_gb']:.1f}GB "
                      f"({sys_info['memory_percent']:.1f}%)")

                # 写入CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_time,
                        gpu_info['gpu_utilization'],
                        gpu_info['memory_used_mb'],
                        gpu_info['memory_total_mb'],
                        gpu_info['memory_percent'],
                        gpu_info['temperature'],
                        sys_info['cpu_percent'],
                        sys_info['memory_used_gb'],
                        sys_info['memory_total_gb'],
                        sys_info['memory_percent']
                    ])

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

    parser.add_argument('--interval', type=int, default=2,
                       help='监控间隔（秒）')
    parser.add_argument('--duration', type=int, default=0,
                       help='监控时长（秒），0表示无限')
    parser.add_argument('--output', type=str, default=None,
                       help='输出CSV文件路径')

    args = parser.parse_args()

    # 默认输出文件
    if args.output:
        output_file = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'logs'
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"gpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    monitor_loop(args.interval, args.duration, output_file)


if __name__ == '__main__':
    main()
