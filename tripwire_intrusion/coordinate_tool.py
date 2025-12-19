"""
坐标标定工具
通过鼠标点击视频帧获取绊线坐标，生成配置文件
"""

import cv2
import json
import argparse
from pathlib import Path


class CoordinateTool:
    """坐标标定工具"""

    def __init__(self, source, output_path=None):
        """
        Args:
            source: 视频路径或摄像头ID
            output_path: 默认输出路径
        """
        self.source = source
        self.output_path = output_path or "tripwire_config.json"
        self.frame = None
        self.tripwires = []
        self.current_points = []
        self.global_direction = "double-direction"  # 全局方向设置
        self.window_name = 'Coordinate Tool'

    def load_frame(self):
        """加载第一帧"""
        if Path(self.source).exists():
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(int(self.source))

        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")

        ret, self.frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("无法读取视频帧")

        print(f"✓ 加载视频帧: {self.frame.shape[1]}x{self.frame.shape[0]}")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调 - 连续绊线模式"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 如果已有线段，新线段从上一条的终点开始
            if self.tripwires:
                last_tripwire = self.tripwires[-1]
                last_end_point = last_tripwire['points'][1]

                # 创建新线段：起点是上一条的终点，终点是当前点击点
                self.current_points = [last_end_point, [x, y]]
                print(f"线段 {len(self.tripwires) + 1} - 起点: {last_end_point}, 终点: [{x}, {y}]")
                self._add_tripwire()
            else:
                # 第一条线段：需要两个点
                self.current_points.append([x, y])
                print(f"第一条线段 - 点 {len(self.current_points)}: [{x}, {y}]")

                if len(self.current_points) == 2:
                    self._add_tripwire()

    def _add_tripwire(self):
        """添加绊线"""
        tripwire_id = f"line_{len(self.tripwires) + 1}"

        tripwire = {
            "id": tripwire_id,
            "points": self.current_points.copy(),
            "direction": self.global_direction,  # 使用全局方向
            "enabled": True,
            "alert_cooldown": 2.0,
            "color": [0, 255, 0]
        }

        self.tripwires.append(tripwire)
        print(f"✓ 添加线段 {len(self.tripwires)}: {self.current_points[0]} → {self.current_points[1]}\n")

        self.current_points = []

    def _set_direction(self):
        """设置全局方向（应用于所有线段）"""
        if not self.tripwires:
            print("✗ 没有绊线可设置")
            return

        print("\n选择方向（应用于所有线段）:")
        print("  1. left-to-right  (从左到右)")
        print("  2. right-to-left  (从右到左)")
        print("  3. double-direction  (双向)")

        choice = input("请输入 (1/2/3): ").strip()

        direction_map = {
            '1': 'left-to-right',
            '2': 'right-to-left',
            '3': 'double-direction'
        }

        if choice in direction_map:
            self.global_direction = direction_map[choice]

            # 应用到所有线段
            for tw in self.tripwires:
                tw['direction'] = self.global_direction

            print(f"✓ 所有线段方向已设置为: {self.global_direction}\n")
        else:
            print("✗ 无效选择\n")

    def _draw_display(self):
        """绘制显示"""
        display = self.frame.copy()

        # 绘制已完成的绊线
        for i, tw in enumerate(self.tripwires):
            p1 = tuple(tw['points'][0])
            p2 = tuple(tw['points'][1])
            color = tuple(tw['color'])

            # 线段
            cv2.line(display, p1, p2, color, 3, cv2.LINE_AA)

            # 端点
            cv2.circle(display, p1, 6, color, -1)
            cv2.circle(display, p2, 6, color, -1)

            # 标签
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            label = f"{tw['id']} ({tw['direction'][:3]})"
            cv2.putText(display, label, (mid_x, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制当前点
        for pt in self.current_points:
            cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)

        # 绘制下一个连接点（高亮显示）
        if self.tripwires:
            last_end = tuple(self.tripwires[-1]['points'][1])
            cv2.circle(display, last_end, 10, (0, 255, 255), 2)  # 黄色圆圈
            cv2.circle(display, last_end, 3, (0, 255, 255), -1)

        return display

    def run(self):
        """运行标定工具"""
        self.load_frame()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("绊线坐标标定工具 - 连续折线模式")
        print("=" * 60)
        print("使用说明:")
        print("  【绘制折线】")
        print("  1. 点击第一个点，再点击第二个点 → 创建第一条线段")
        print("  2. 继续点击下一个点 → 自动从上一个点延续，创建折线")
        print("  3. 所有线段共享同一个方向设置")
        print("")
        print("  【快捷键】")
        print("  - 'd': 设置整体方向 (左→右/右→左/双向)")
        print("  - 's': 保存配置到JSON文件")
        print("  - 'r': 重置所有线段，重新开始")
        print("  - 'q': 退出工具")
        print("")
        print("  【提示】")
        print("  - 黄色圆圈标记下一条线段的起点")
        print("  - 适合绘制围栏、边界等连续区域")
        print("=" * 60 + "\n")

        while True:
            display = self._draw_display()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n退出")
                break

            elif key == ord('r'):
                self.tripwires = []
                self.current_points = []
                print("\n✓ 已重置所有绊线\n")

            elif key == ord('d'):
                self._set_direction()

            elif key == ord('s'):
                self._save_config()

        cv2.destroyAllWindows()

    def _save_config(self):
        """保存配置"""
        if not self.tripwires:
            print("\n✗ 没有绊线可保存\n")
            return

        # 询问是否使用默认路径
        print(f"\n默认保存路径: {self.output_path}")
        user_input = input("按Enter使用默认路径，或输入新路径: ").strip()

        if user_input:
            output_path = user_input
        else:
            output_path = self.output_path

        config = {"tripwires": self.tripwires}

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 配置已保存到: {output_file.absolute()}")
        print(f"  连续折线: 共 {len(self.tripwires)} 条线段")
        print(f"  整体方向: {self.global_direction}")
        print("")

        # 显示每条线段的信息
        direction_cn = {
            'left-to-right': '左→右',
            'right-to-left': '右→左',
            'double-direction': '双向'
        }.get(self.global_direction, self.global_direction)

        print(f"  线段明细:")
        for i, tw in enumerate(self.tripwires, 1):
            print(f"    {i}. {tw['points'][0]} → {tw['points'][1]}")

        print(f"\n  方向规则: {direction_cn}")
        print(f"\n下一步: 使用此配置运行检测")
        print(f"  python tripwire_intrusion/tripwire_detector.py \\")
        print(f"      --source <your_video.mp4> \\")
        print(f"      --weights <your_model.pt> \\")
        print(f"      --config {output_file} \\")
        print(f"      --tracker bytetrack \\")
        print(f"      --show --save\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='绊线坐标标定工具')
    parser.add_argument('--source', type=str, required=True,
                       help='视频路径或摄像头ID (0, 1, ...)')
    parser.add_argument('--output', type=str, default='tripwire_intrusion/config_line.json',
                       help='输出配置文件路径 (默认: tripwire_config.json)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    tool = CoordinateTool(args.source, args.output)
    tool.run()


if __name__ == '__main__':
    main()
