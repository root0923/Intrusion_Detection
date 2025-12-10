"""
绊线入侵检测主程序
使用Ultralytics内置跟踪器（ByteTrack/BoT-SORT）
支持视频文件、RTSP流、摄像头输入
"""

import sys
import cv2
import argparse
import requests
import base64
from pathlib import Path
from datetime import datetime
import time
from collections import deque
from typing import Tuple, Any
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from tripwire_intrusion.tripwire_monitor import TripwireMonitor
from tripwire_intrusion.visualizer import TripwireVisualizer
import warnings
warnings.filterwarnings("ignore")

class Track:
    """轨迹对象（适配Ultralytics跟踪结果）"""

    def __init__(self, track_id: int, bbox: list, conf: float, cls: int):
        self.track_id = track_id
        self.bbox = bbox
        self.conf = conf
        self.cls = cls
        self.trajectory = deque(maxlen=30)

        # 添加底部中心点到轨迹
        center = self._get_bottom_center(bbox)
        self.trajectory.append(center)

    @staticmethod
    def _get_bottom_center(bbox: list) -> Tuple[float, float]:
        """获取检测框底部中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def get_latest_position(self) -> Tuple[float, float]:
        """获取最新位置"""
        return self.trajectory[-1] if self.trajectory else (0, 0)


class TripwireDetectionSystem:
    """绊线检测系统（使用Ultralytics跟踪器）"""

    def __init__(self, args):
        """
        Args:
            args: 命令行参数
        """
        self.args = args

        # 1. 初始化YOLO模型
        print("\n" + "=" * 60)
        print("初始化YOLO模型...")
        print("=" * 60)

        # 使用 yaml_model_load 加载模型配置
        print(f"✓ 加载模型配置: {args.model_yaml}")
        yaml_dict = yaml_model_load(args.model_yaml)

        # 从 YAML 文件读取 ch（强制使用YAML的配置）
        self.model_ch = yaml_dict.get('ch', 3)
        print(f"  从YAML文件读取: ch={self.model_ch}")

        # 从权重文件中读取 nc（类别数）
        print(f"✓ 加载权重文件: {args.weights}")
        ckpt = torch.load(args.weights, map_location='cpu')

        if 'model' in ckpt and hasattr(ckpt['model'], 'yaml'):
            nc = ckpt['model'].yaml.get('nc', 80)
            print(f"  从权重文件读取: nc={nc}")
        else:
            nc = yaml_dict.get('nc', 80)
            print(f"  从YAML文件读取: nc={nc}")

        # 方法1：直接从权重文件加载（推荐，最简单）
        self.model = YOLO(args.weights)
        print(f"✓ 模型加载完成（直接从权重文件）")

        # 验证模型配置
        print(f"  模型第一层输入通道: {list(self.model.model.model.children())[0].conv.in_channels}")

        print(f"✓ 跟踪器: {args.tracker.upper()}")

        # 2. 初始化绊线监控器
        print("\n" + "=" * 60)
        print("初始化绊线监控器...")
        print("=" * 60)
        self.repeat_alarm_interval = float(getattr(args, 'repeat_alarm_interval', 30.0))
        self.monitor = TripwireMonitor(
            args.config,
            global_cooldown=self.repeat_alarm_interval  # 传入全局冷却时间
        )

        # 3. 初始化可视化器
        self.visualizer = TripwireVisualizer(
            tripwires=self.monitor.get_tripwires(),
            draw_trajectory=args.draw_trajectory,
            trajectory_length=args.trajectory_length
        )

        # 类别名称
        self.class_names = {0: 'person'}

        # 轨迹缓存（用于保持历史轨迹）
        self.track_history = {}  # {track_id: Track对象}
        self.track_last_seen = {}  # {track_id: frame_number} 记录每个track最后出现的帧

        # 内存管理参数
        self.max_frames_to_keep = 60  # 保留最近60帧未出现的track（约2秒@30fps）

        # 统计信息
        self.frame_count = 0
        self.total_events = 0
        self.fps_list = []
        # 报警/上传配置
        self.alarm_url = getattr(args, 'alarm_url', None)
        self.save_width = int(getattr(args, 'save_width', 0)) if getattr(args, 'save_width', None) else None
        self.save_height = int(getattr(args, 'save_height', 0)) if getattr(args, 'save_height', None) else None
        # 每秒处理帧数（抽帧策略）
        self.process_fps = float(getattr(args, 'process_fps', 2.0))
        # 重复报警间隔（秒），由 TripwireMonitor 内部管理
        self.repeat_alarm_interval = float(getattr(args, 'repeat_alarm_interval', 30.0))

        # 缓存上一次可视化帧（用于抽帧时显示）
        self._last_vis_frame = None
        # 报警图片本地保存目录（输出目录下的 alarms 子目录）
        try:
            output_dir = Path(getattr(args, 'output_dir', 'runs/tripwire'))
        except Exception:
            output_dir = Path('runs/tripwire')
        self.alarm_dir = output_dir / 'alarms'
        self.alarm_dir.mkdir(parents=True, exist_ok=True)
        print(f"报警图片目录: {self.alarm_dir}")

    def _update_tracks(self, results):
        """
        将Ultralytics跟踪结果转换为Track对象并更新轨迹历史

        Args:
            results: YOLO跟踪结果

        Returns:
            List[Track]: 当前帧的轨迹列表
        """
        current_tracks = []
        current_track_ids = set()

        # 检查是否有检测结果
        if results[0].boxes is None or len(results[0].boxes) == 0:
            self._cleanup_old_tracks()
            return current_tracks

        # 检查是否有跟踪ID
        if results[0].boxes.id is None:
            self._cleanup_old_tracks()
            return current_tracks

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        # 更新轨迹
        for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
            current_track_ids.add(track_id)

            # 如果是新轨迹，创建Track对象
            if track_id not in self.track_history:
                self.track_history[track_id] = Track(
                    track_id, box.tolist(), float(conf), int(cls)
                )
            else:
                # 更新现有轨迹
                track = self.track_history[track_id]
                track.bbox = box.tolist()
                track.conf = float(conf)
                track.cls = int(cls)

                # 添加新位置到轨迹
                center = track._get_bottom_center(box.tolist())
                track.trajectory.append(center)

            # 更新最后出现时间
            self.track_last_seen[track_id] = self.frame_count

            current_tracks.append(self.track_history[track_id])

        # 清理旧的track
        self._cleanup_old_tracks()

        return current_tracks

    def _cleanup_old_tracks(self):
        """清理长时间未出现的track，防止内存泄漏"""
        tracks_to_remove = []

        for track_id, last_seen in self.track_last_seen.items():
            # 如果超过max_frames_to_keep帧未出现，标记删除
            if self.frame_count - last_seen > self.max_frames_to_keep:
                tracks_to_remove.append(track_id)

        # 删除旧track
        for track_id in tracks_to_remove:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_last_seen:
                del self.track_last_seen[track_id]

        # 可选：打印清理信息（调试用）
        if tracks_to_remove:
            print(f"[内存管理] 清理了 {len(tracks_to_remove)} 个旧track，当前活跃: {len(self.track_history)}")

    def resize_and_encode_image(self, image: Any) -> Tuple[Any, str]:
        """
        将图片缩放到 self.save_width x self.save_height（如果设置），并返回 (resized_image, base64_str)
        """
        img = image
        if self.save_width and self.save_height:
            try:
                resized = cv2.resize(img, (int(self.save_width), int(self.save_height)), interpolation=cv2.INTER_AREA)
            except Exception:
                resized = img
        else:
            resized = img

        success, buffer = cv2.imencode('.jpg', resized)
        if not success:
            success, buffer = cv2.imencode('.jpg', img)

        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return resized, img_b64

    def _maybe_send_alarm(self, event, vis_frame):
        """
        发送报警到配置的 URL
        冷却逻辑已由 TripwireMonitor 内部管理，这里直接发送
        """
        now = time.time()

        # 生成图片 base64（并获取缩放后的图片对象）
        resized_img, img_b64 = self.resize_and_encode_image(vis_frame)

        # 保存报警图片到本地文件（使用毫秒时间戳和绊线ID）
        try:
            ts_ms = int(now * 1000)
            alarm_filename = self.alarm_dir / f"alarm_{ts_ms}_{event.tripwire_id}.jpg"
            cv2.imwrite(str(alarm_filename), resized_img)
            print(f"✓ 报警图片已保存: {alarm_filename}")
        except Exception as e:
            print(f"✗ 保存报警图片失败: {e}")

        # 上传的报警信息仅包含时间和图片 base64 编码
        alarm_payload = {
            'timestamp': now,
            'time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
            'image': img_b64
        }

        if not self.alarm_url:
            print("✗ 未配置报警URL，跳过发送报警")
            return

        try:
            resp = requests.post(self.alarm_url, json=alarm_payload, timeout=5)
            if resp.status_code == 200:
                print(f"✓ 报警已发送: track={event.track_id}, tripwire={event.tripwire_id}")
            else:
                print(f"✗ 报警发送失败: HTTP {resp.status_code}")
        except Exception as e:
            print(f"✗ 报警发送异常: {e}")

    def process_frame(self, frame):
        """
        处理单帧

        Args:
            frame: 视频帧

        Returns:
            vis_frame: 可视化后的帧
            num_events: 本帧事件数
        """
        start_time = time.time()


        # 1. YOLO检测+跟踪（一步完成）

        results = self.model.track(
            frame,
            conf=self.args.conf_thresh,
            iou=self.args.nms_iou,
            imgsz=self.args.imgsz,
            use_simotm="RGB",
            channels=3,
            persist=True,  # 保持轨迹ID
            tracker=f"{self.args.tracker}.yaml",  # 跟踪器配置
            verbose=False,
            device=self.args.device
        )

        # 2. 转换为Track对象
        tracks = self._update_tracks(results)

        # 3. 绊线监控
        events = self.monitor.update(tracks)

        # 4. 可视化
        vis_frame = self.visualizer.draw(
            frame,
            tracks=tracks,
            recent_events=events,
            class_names=self.class_names
        )

        # 5. 统计
        self.frame_count += 1
        self.total_events += len(events)
        fps = 1.0 / (time.time() - start_time)
        self.fps_list.append(fps)

        # 绘制FPS
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (vis_frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 返回事件对象列表以便外层发送报警
        return vis_frame, events

    def run_video(self, video_path):
        """
        处理视频文件

        Args:
            video_path: 视频路径
        """
        print("\n" + "=" * 60)
        print(f"处理视频: {video_path}")
        print("=" * 60)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {video_path}")
            return

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")

        # 设置图像高度（用于方向判断的坐标系转换）
        self.monitor.set_image_height(height)

        # 视频写入器
        out = None
        if self.args.save:
            output_path = Path(self.args.output_dir) / f"{Path(video_path).stem}_tripwire.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"输出视频: {output_path}")

        print("\n开始处理... (按 'q' 退出)\n")

        # 计算抽帧间隔
        if fps and self.process_fps and self.process_fps > 0:
            process_interval = max(1, int(round(float(fps) / float(self.process_fps))))
        else:
            process_interval = 1

        print(f"抽帧设置: 每 {process_interval} 帧处理一次 (目标处理 {self.process_fps} 帧/s)")

        input_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_frame_idx += 1

            # 决定是否在本帧运行检测（抽帧）
            process_this_frame = ((input_frame_idx - 1) % process_interval) == 0

            if process_this_frame:
                # 处理帧（包含检测、跟踪、可视化）
                vis_frame, events = self.process_frame(frame)
                # 缓存最新可视化帧
                self._last_vis_frame = vis_frame

                # 对每个事件尝试发送报警
                if events and len(events) > 0:
                    for ev in events:
                        self._maybe_send_alarm(ev, vis_frame)
            else:
                # 非处理帧：使用上一次可视化帧或简单绘制绊线以减少开销
                if self._last_vis_frame is not None:
                    vis_frame = self._last_vis_frame
                else:
                    vis_frame = self.visualizer.draw_tripwire_setup(frame)

            # 显示
            if self.args.show:
                cv2.imshow('Tripwire Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 保存
            if out is not None:
                out.write(vis_frame)

            # 进度（按输入帧数汇报）
            if input_frame_idx % 30 == 0:
                progress = input_frame_idx / total_frames * 100 if total_frames else 0
                avg_fps = sum(self.fps_list[-30:]) / min(30, len(self.fps_list)) if self.fps_list else 0
                print(f"进度: {input_frame_idx}/{total_frames} ({progress:.1f}%), "
                      f"FPS: {avg_fps:.1f}, 总事件: {self.total_events}")

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        self._print_summary()

    def _print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 60)
        print("检测完成")
        print("=" * 60)
        print(f"总帧数: {self.frame_count}")
        print(f"总事件: {self.total_events}")
        if self.fps_list:
            print(f"平均FPS: {sum(self.fps_list) / len(self.fps_list):.2f}")

        # 导出事件
        if self.args.export_events:
            event_path = Path(self.args.output_dir) / "events.json"
            self.monitor.export_events(str(event_path))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='绊线入侵检测系统 (使用Ultralytics跟踪器)')

    # 模型参数
    parser.add_argument('--model-yaml', type=str, default="ultralytics/cfg/models/11/yolo11x.yaml",
                       help='模型配置YAML文件')
    parser.add_argument('--weights', type=str,
                       default='data/LLVIP_IF-yolo11x-e300-16-pretrained.pt',
                       help='模型权重文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0 或 cpu)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='推理图像尺寸')

    # 检测参数
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--nms-iou', type=float, default=0.7,
                       help='NMS IoU阈值')

    # 跟踪参数
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'botsort'],
                       help='跟踪器类型: bytetrack(更快) 或 botsort(更准)')

    # 绊线配置
    parser.add_argument('--config', type=str,
                       default='tripwire_intrusion/config_line.json',
                       help='绊线配置文件')

    # 输入输出
    parser.add_argument('--source', type=str, default='data/dataset/video_IR/INO_TreesAndRunner_T.avi',
                       help='视频路径、摄像头ID(0,1,...)或RTSP地址')
    parser.add_argument('--output-dir', type=str, default='runs/tripwire',
                       help='输出目录')
    parser.add_argument('--save', action='store_true',
                       help='保存输出视频')
    parser.add_argument('--show', action='store_true',
                       help='显示实时结果')
    parser.add_argument('--export-events', action='store_true',
                       help='导出事件到JSON')

    # 可视化参数
    parser.add_argument('--draw-trajectory', action='store_true', default=True,
                       help='绘制轨迹')
    parser.add_argument('--trajectory-length', type=int, default=30,
                       help='轨迹显示长度')

    # 报警与保存参数
    parser.add_argument('--alarm-url', type=str, default=None,
                       help='报警接口URL (POST JSON)')
    parser.add_argument('--save-width', type=int, default=1280,
                       help='报警图片保存/上传宽度')
    parser.add_argument('--save-height', type=int, default=720,
                       help='报警图片保存/上传高度')
    parser.add_argument('--process-fps', type=float, default=5.0,
                       help='每秒处理帧数（抽帧），例如2表示每秒处理2帧）')
    parser.add_argument('--repeat-alarm-interval', type=float, default=30.0,
                       help='重复报警最小间隔（秒）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建检测系统
    system = TripwireDetectionSystem(args)

    # 判断输入类型
    source = args.source

    # 视频文件
    if Path(source).exists() and Path(source).is_file():
        system.run_video(source)

    # 摄像头ID或RTSP
    else:
        system.run_camera(source)


if __name__ == '__main__':
    main()
