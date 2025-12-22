"""
实时性能可视化工具 - Real-time Performance Visualization

功能：
- 从日志文件解析性能数据
- 实时绘制推理时间、帧间隔曲线
- 可视化延迟情况
- 支持多路对比

使用方法：
    python visualize_performance.py [日志文件路径]
    或者实时监控：tail -f detector.log | python visualize_performance.py --stdin
"""

import re
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict, deque
import argparse


class PerformanceVisualizer:
    """性能可视化器"""

    def __init__(self, max_points=100):
        """
        Args:
            max_points: 每个图表显示的最大数据点数
        """
        self.max_points = max_points

        # 数据存储：{camera_key: {'timestamps': [], 'inference': [], 'interval': [], ...}}
        self.data = defaultdict(lambda: {
            'timestamps': deque(maxlen=max_points),
            'inference': deque(maxlen=max_points),
            'rules': deque(maxlen=max_points),
            'total': deque(maxlen=max_points),
            'interval': deque(maxlen=max_points),
            'expected_interval': None,
            'delay_pct': deque(maxlen=max_points),
        })

        # 创建图表
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('入侵检测系统 - 实时性能监控', fontsize=16, fontweight='bold')

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 子图标题
        self.axes[0].set_title('推理时间 (Inference Time)')
        self.axes[0].set_ylabel('时间 (ms)')
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].set_title('帧间隔 vs 期望间隔 (Frame Interval)')
        self.axes[1].set_ylabel('间隔 (ms)')
        self.axes[1].grid(True, alpha=0.3)

        self.axes[2].set_title('延迟百分比 (Delay Percentage)')
        self.axes[2].set_ylabel('延迟 (%)')
        self.axes[2].set_xlabel('样本点')
        self.axes[2].grid(True, alpha=0.3)
        self.axes[2].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='无延迟')
        self.axes[2].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='轻微延迟阈值')
        self.axes[2].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='严重延迟阈值')

        # 线条存储
        self.lines = {}

        # 颜色映射
        self.colors = plt.cm.tab10.colors

    def parse_log_line(self, line):
        """
        解析日志行

        示例：
        [2025-12-22 10:20:29] [INFO] [camera_01] 帧: 750 | 推理: 72.3ms | 规则: 0.1ms |
        总计: 72.4ms | 帧间隔: 205.3ms (期望: 200.0ms, 延迟: +2.6%) ✓ | 活跃规则: [...]
        """
        # 提取摄像头key
        camera_match = re.search(r'\[([^\]]+_\d+)\]', line)
        if not camera_match:
            return None

        camera_key = camera_match.group(1)

        # 提取性能数据
        inference_match = re.search(r'推理:\s*([\d.]+)ms', line)
        rules_match = re.search(r'规则:\s*([\d.]+)ms', line)
        total_match = re.search(r'总计:\s*([\d.]+)ms', line)
        interval_match = re.search(r'帧间隔:\s*([\d.]+)ms', line)
        expected_match = re.search(r'期望:\s*([\d.]+)ms', line)
        delay_match = re.search(r'延迟:\s*([+-]?[\d.]+)%', line)

        if not all([inference_match, total_match]):
            return None

        result = {
            'camera_key': camera_key,
            'inference': float(inference_match.group(1)),
            'rules': float(rules_match.group(1)) if rules_match else 0.0,
            'total': float(total_match.group(1)),
        }

        if interval_match:
            result['interval'] = float(interval_match.group(1))
        if expected_match:
            result['expected_interval'] = float(expected_match.group(1))
        if delay_match:
            result['delay_pct'] = float(delay_match.group(1))

        return result

    def update_data(self, data_point):
        """更新数据点"""
        camera_key = data_point['camera_key']
        cam_data = self.data[camera_key]

        # 添加时间戳（使用样本索引）
        cam_data['timestamps'].append(len(cam_data['timestamps']))

        # 添加性能数据
        cam_data['inference'].append(data_point['inference'])
        cam_data['rules'].append(data_point['rules'])
        cam_data['total'].append(data_point['total'])

        if 'interval' in data_point:
            cam_data['interval'].append(data_point['interval'])
        if 'expected_interval' in data_point:
            cam_data['expected_interval'] = data_point['expected_interval']
        if 'delay_pct' in data_point:
            cam_data['delay_pct'].append(data_point['delay_pct'])

    def update_plot(self, _frame=None):
        """更新图表"""
        # 清空所有子图
        for ax in self.axes:
            ax.clear()

        # 重新设置基本属性
        self.axes[0].set_title('推理时间 (Inference Time)', fontsize=12, fontweight='bold')
        self.axes[0].set_ylabel('时间 (ms)')
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].set_title('帧间隔 vs 期望间隔 (Frame Interval)', fontsize=12, fontweight='bold')
        self.axes[1].set_ylabel('间隔 (ms)')
        self.axes[1].grid(True, alpha=0.3)

        self.axes[2].set_title('延迟百分比 (Delay Percentage)', fontsize=12, fontweight='bold')
        self.axes[2].set_ylabel('延迟 (%)')
        self.axes[2].set_xlabel('样本点')
        self.axes[2].grid(True, alpha=0.3)
        self.axes[2].axhline(y=0, color='g', linestyle='--', alpha=0.5, linewidth=1)
        self.axes[2].axhline(y=20, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        self.axes[2].axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=1)

        # 绘制每个摄像头的数据
        for idx, (camera_key, cam_data) in enumerate(self.data.items()):
            if len(cam_data['timestamps']) == 0:
                continue

            color = self.colors[idx % len(self.colors)]
            x = list(cam_data['timestamps'])

            # 子图1: 推理时间
            if len(cam_data['inference']) > 0:
                self.axes[0].plot(x, list(cam_data['inference']),
                                 label=f'{camera_key} (推理)',
                                 color=color, linewidth=2, marker='o', markersize=3)
                self.axes[0].plot(x, list(cam_data['total']),
                                 label=f'{camera_key} (总计)',
                                 color=color, linewidth=1, linestyle='--', alpha=0.6)

            # 子图2: 帧间隔
            if len(cam_data['interval']) > 0:
                self.axes[1].plot(x[-len(cam_data['interval']):], list(cam_data['interval']),
                                 label=f'{camera_key} (实际)',
                                 color=color, linewidth=2, marker='o', markersize=3)

                # 绘制期望间隔
                if cam_data['expected_interval'] is not None:
                    expected_line = [cam_data['expected_interval']] * len(x[-len(cam_data['interval']):])
                    self.axes[1].plot(x[-len(cam_data['interval']):], expected_line,
                                     label=f'{camera_key} (期望)',
                                     color=color, linewidth=1, linestyle=':', alpha=0.8)

            # 子图3: 延迟百分比
            if len(cam_data['delay_pct']) > 0:
                delays = list(cam_data['delay_pct'])
                x_delay = x[-len(delays):]

                # 根据延迟程度填充不同颜色
                colors_fill = []
                for d in delays:
                    if d > 50:
                        colors_fill.append('red')
                    elif d > 20:
                        colors_fill.append('orange')
                    elif d > 0:
                        colors_fill.append('yellow')
                    else:
                        colors_fill.append('green')

                self.axes[2].bar(x_delay, delays,
                                label=f'{camera_key}',
                                color=colors_fill, alpha=0.7, width=0.8)

        # 添加图例
        for ax in self.axes:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

        plt.tight_layout()

    def process_log_file(self, log_file):
        """处理日志文件（批量）"""
        print(f"正在读取日志文件: {log_file}")

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                data_point = self.parse_log_line(line)
                if data_point:
                    self.update_data(data_point)

        print(f"数据加载完成，共 {len(self.data)} 个摄像头")
        self.update_plot()
        plt.show()

    def process_stdin(self):
        """处理标准输入（实时）"""
        print("等待日志输入... (Ctrl+C 停止)")

        def update_from_stdin(_frame):
            try:
                line = sys.stdin.readline()
                if line:
                    data_point = self.parse_log_line(line)
                    if data_point:
                        self.update_data(data_point)
                        self.update_plot()
            except:
                pass

        # 创建动画（需要保持引用以防止垃圾回收）
        _animation = animation.FuncAnimation(self.fig, update_from_stdin,
                                     interval=1000, blit=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='实时性能可视化工具')
    parser.add_argument('log_file', nargs='?', help='日志文件路径')
    parser.add_argument('--stdin', action='store_true', help='从标准输入读取（实时监控）')
    parser.add_argument('--max-points', type=int, default=100, help='最大显示数据点数')

    args = parser.parse_args()

    visualizer = PerformanceVisualizer(max_points=args.max_points)

    if args.stdin:
        visualizer.process_stdin()
    elif args.log_file:
        visualizer.process_log_file(args.log_file)
    else:
        print("错误: 请指定日志文件或使用 --stdin 选项")
        print("示例:")
        print("  python visualize_performance.py detector.log")
        print("  tail -f detector.log | python visualize_performance.py --stdin")
        sys.exit(1)


if __name__ == '__main__':
    main()
