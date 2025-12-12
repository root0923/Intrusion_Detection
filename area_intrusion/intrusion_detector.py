"""
区域入侵检测系统
支持：红外视频ROI区域入侵判断、报警机制
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import json
import time
import requests
import base64
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from detector import Detector
import warnings
warnings.filterwarnings("ignore")


class ROIManager:
    """ROI区域管理器（使用mask方式）"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: ROI配置JSON文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.image_width = config.get('image_width', 640)
        self.image_height = config.get('image_height', 480)
        self.rois = [np.array(roi, dtype=np.int32) for roi in config['rois']]

        # 创建合并所有ROI的总mask
        self.combined_mask = self._create_combined_mask()

        print(f"✓ 加载ROI配置: {len(self.rois)} 个区域")

    def _create_combined_mask(self) -> np.ndarray:
        """
        创建包含所有ROI的合并mask（所有ROI区域内为255，区域外为0）

        Returns:
            np.ndarray: 合并后的mask
        """
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for roi in self.rois:
            cv2.fillPoly(mask, [roi], 255)
        return mask

    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """
        将所有ROI区域外的像素变黑（只保留ROI区域内的图像）

        Args:
            image: 输入图像

        Returns:
            masked_image: ROI区域外变黑后的图像
        """
        # 调整mask尺寸以匹配输入图像
        if image.shape[:2] != (self.image_height, self.image_width):
            mask = cv2.resize(self.combined_mask,
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        else:
            mask = self.combined_mask

        # 应用mask：将所有ROI外的区域变黑
        masked_image = image.copy()
        masked_image[mask == 0] = 0

        return masked_image

    def draw_rois(self, image: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        在图像上绘制ROI区域

        Args:
            image: 输入图像
            color: ROI边界颜色
            thickness: 线条粗细

        Returns:
            绘制后的图像
        """
        img_draw = image.copy()

        for roi_id, roi in enumerate(self.rois):
            # 绘制多边形边界
            cv2.polylines(img_draw, [roi], isClosed=True, color=color, thickness=thickness)

            # 绘制半透明填充
            overlay = img_draw.copy()
            cv2.fillPoly(overlay, [roi], color)
            cv2.addWeighted(overlay, 0.2, img_draw, 0.8, 0, img_draw)

            # 添加ROI标签
            centroid = roi.mean(axis=0).astype(int)
            cv2.putText(img_draw, f'ROI-{roi_id}', tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img_draw


class AlarmManager:
    """报警管理器"""

    def __init__(self,
                 conf_threshold: float = 0.25,
                 first_alarm_duration: float = 1.0,
                 repeat_alarm_interval: float = 30.0,
                 tolerance_time: float = 3.0,
                 alarm_url: Optional[str] = None,
                 save_height: Optional[int] = None,
                 save_width: Optional[int] = None):
        """
        Args:
            conf_threshold: 置信度阈值
            first_alarm_duration: 首次报警时间（秒），目标持续出现该时长才报警
            repeat_alarm_interval: 重复报警间隔（秒），该时间段内只报警一次
            tolerance_time: 容忍时间（秒），检测不到目标后的宽限期，防止短暂丢失导致状态重置
            alarm_url: 报警接口URL
        """
        self.conf_threshold = conf_threshold
        self.first_alarm_duration = first_alarm_duration
        self.repeat_alarm_interval = repeat_alarm_interval
        self.tolerance_time = tolerance_time
        self.alarm_url = alarm_url
        # 保存/编码报警图片的目标尺寸
        self.save_height = save_height
        self.save_width = save_width

        # 入侵状态
        self.intrusion_state = {
            'first_time': None,        # 首次检测到目标的时间
            'last_alarm_time': None,   # 上次报警的时间
            'last_seen_time': None     # 最后一次看到目标的时间（容忍机制关键）
        }

        print(f"✓ 报警管理器初始化:")
        print(f"  - 置信度阈值: {conf_threshold}")
        print(f"  - 首次报警时间: {first_alarm_duration}s")
        print(f"  - 重复报警间隔: {repeat_alarm_interval}s")
        print(f"  - 容忍时间: {tolerance_time}s")
        if alarm_url:
            print(f"  - 报警接口: {alarm_url}")

    def update_intrusion(self, detections: List[Dict],
                        frame: np.ndarray) -> List[Dict]:
        """
        更新入侵状态并触发报警（带容忍机制）

        Args:
            detections: 检测结果列表
            frame: 当前帧图像（用于截图）

        Returns:
            List[Dict]: 触发的报警列表
        """
        current_time = time.time()
        alarms = []

        # 过滤置信度，获取高置信度检测
        valid_detections = [det for det in detections if det['conf'] >= self.conf_threshold]

        if valid_detections:
            # 检测到目标
            if self.intrusion_state['first_time'] is None:
                # 首次检测到，记录开始时间
                self.intrusion_state['first_time'] = current_time
                self.intrusion_state['last_seen_time'] = current_time
                print(f"[入侵检测] 检测到入侵 (消抖中...)")
            else:
                # 持续检测到目标，更新最后看到时间
                self.intrusion_state['last_seen_time'] = current_time

                # 检查是否达到报警条件
                duration = current_time - self.intrusion_state['first_time']

                # 条件1：持续时间超过首次报警时间（消抖）
                if duration >= self.first_alarm_duration:
                    # 条件2：距离上次报警超过重复报警间隔
                    if (self.intrusion_state['last_alarm_time'] is None or
                        current_time - self.intrusion_state['last_alarm_time'] >= self.repeat_alarm_interval):

                        # 触发报警
                        alarm = self._create_alarm()
                        alarms.append(alarm)

                        # 更新最后报警时间
                        self.intrusion_state['last_alarm_time'] = current_time

                        print(f"[入侵检测] 🚨 报警触发! (持续 {duration:.1f}s, 检测数: {len(valid_detections)})")
        else:
            # 当前帧未检测到目标
            if self.intrusion_state['first_time'] is not None:
                # 之前有入侵状态，检查是否超过容忍时间
                if self.intrusion_state['last_seen_time'] is not None:
                    gap = current_time - self.intrusion_state['last_seen_time']

                    if gap > self.tolerance_time:
                        # 超过容忍时间，认为入侵真正结束
                        duration = self.intrusion_state['last_seen_time'] - self.intrusion_state['first_time']
                        print(f"[入侵检测] 入侵结束 (持续 {duration:.1f}s, 容忍期后确认)")
                        self.intrusion_state['first_time'] = None
                        # 注意：不重置 last_alarm_time，以避免频繁报警（保持全局报警间隔限制）
                        self.intrusion_state['last_seen_time'] = None
                    else:
                        # 在容忍时间内，保持状态不变
                        print(f"[入侵检测] 暂时未检测到目标 (容忍中: {gap:.1f}s / {self.tolerance_time}s)")

        return alarms

    def _create_alarm(self) -> Dict:
        """
        创建报警信息（只包含时间戳，图片将在可视化后添加）

        Args:
            detections: 检测结果
            frame: 当前帧

        Returns:
            Dict: 报警信息 {'timestamp': str, 'image': str (稍后添加)}
        """
        alarm = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return alarm

    def _send_alarm(self, alarm: Dict):
        """
        发送报警到接口

        Args:
            alarm: 报警信息
        """
        try:
            response = requests.post(
                self.alarm_url,
                json=alarm,
                timeout=5
            )
            if response.status_code == 200:
                print(f"  ✓ 报警已发送到接口")
            else:
                print(f"  ✗ 报警发送失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ✗ 报警发送异常: {e}")

    def resize_and_encode(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        将图片缩放到 `save_width` x `save_height` 并返回 (resized_image, base64_str)

        如果 `save_height` 或 `save_width` 为 None 或与原图相同，则不缩放。
        """
        img = image

        if not self.save_height or not self.save_width:
            resized = img
        else:
            try:
                resized = cv2.resize(img, (int(self.save_width), int(self.save_height)), interpolation=cv2.INTER_AREA)
            except Exception:
                resized = img

        success, buffer = cv2.imencode('.jpg', resized)
        if not success:
            success, buffer = cv2.imencode('.jpg', img)

        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return resized, image_base64


class IntrusionDetectionSystem:
    """区域入侵检测系统"""

    def __init__(self,
                 detector: Detector,
                 roi_config_path: str,
                 conf_threshold: float = 0.5,
                 first_alarm_duration: float = 2.0,
                 repeat_alarm_interval: float = 30.0,
                 tolerance_time: float = 3.0,
                 save_height: int = 480,
                 save_width: int = 640,
                 target_size: int = 640,
                 process_fps: float = 2.0,
                 alarm_url: Optional[str] = None):
        """
        Args:
            detector: YOLO检测器实例
            roi_config_path: ROI配置文件路径
            conf_threshold: 置信度阈值
            first_alarm_duration: 首次报警时间（秒）
            repeat_alarm_interval: 重复报警间隔（秒）
            tolerance_time: 容忍时间（秒），检测不到目标后的宽限期
            save_height: 保存报警的图片高度
            save_width: 保存报警的图片宽度
            target_size: YOLO检测输入/目标尺寸
            process_fps: 每秒处理帧数（抽帧）
            alarm_url: 报警接口URL
        """
        self.detector = detector
        self.roi_manager = ROIManager(roi_config_path)
        self.alarm_manager = AlarmManager(
            conf_threshold=conf_threshold,
            first_alarm_duration=first_alarm_duration,
            repeat_alarm_interval=repeat_alarm_interval,
            tolerance_time=tolerance_time,
            save_height=save_height,
            save_width=save_width,
            alarm_url=alarm_url
        )

        self.target_size = int(target_size)
        self.process_fps = float(process_fps) if process_fps and float(process_fps) > 0 else 2.0

        self.class_names = {0: 'person'}
        self.total_alarms = 0 

    def process_video(self,
                     video_path: str,
                     output_path: Optional[str] = None,
                     display: bool = True,
                     save_alarms: bool = True):
        """
        处理视频并进行区域入侵检测（使用ROI mask方式）

        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径（可选）
            display: 是否实时显示
            save_alarms: 是否保存报警截图
        """
        print(f"\n{'='*60}")
        print(f"开始处理视频: {video_path}")
        print(f"{'='*60}\n")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {video_path}")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

        self._process_stream(cap, fps, width, height, output_path, display, save_alarms, total_frames)

    def run_camera(self, source: str, output_path: Optional[str] = None,
                   display: bool = True, save_alarms: bool = True):
        """
        处理摄像头或RTSP流

        Args:
            source: 摄像头ID(0,1,...)或RTSP地址
            output_path: 输出视频路径（可选）
            display: 是否实时显示
            save_alarms: 是否保存报警截图
        """
        print(f"\n{'='*60}")
        print(f"处理流媒体: {source}")
        print(f"{'='*60}\n")

        # 尝试打开摄像头或RTSP流
        try:
            # 尝试作为摄像头ID
            camera_id = int(source)
            cap = cv2.VideoCapture(camera_id)
        except ValueError:
            # 作为RTSP地址或文件路径
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"✗ 无法打开流媒体源: {source}")
            return

        # 获取流信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # RTSP流可能返回0，默认30fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"流媒体信息: {width}x{height} @ {fps}fps")
        print("注意: 流媒体没有总帧数限制，按 'q' 退出\n")

        self._process_stream(cap, fps, width, height, output_path, display, save_alarms, total_frames=None)

    def _process_stream(self, cap, fps: int, width: int, height: int,
                       output_path: Optional[str], display: bool,
                       save_alarms: bool, total_frames: Optional[int]):
        """
        通用流处理方法

        Args:
            cap: cv2.VideoCapture对象
            fps: 帧率
            width: 宽度
            height: 高度
            output_path: 输出路径
            display: 是否显示
            save_alarms: 是否保存报警
            total_frames: 总帧数（流媒体为None）
        """

        # 初始化输出视频
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出视频: {output_path}")

        # 创建报警截图目录
        alarm_dir = None
        if save_alarms:
            alarm_dir = Path(output_path).parent / 'alarms' if output_path else Path('runs/alarms')
            alarm_dir.mkdir(parents=True, exist_ok=True)
            print(f"报警截图目录: {alarm_dir}")

        # 处理视频帧
        frame_count = 0
        start_time = time.time()
        total_alarms = 0

        # 计算抽帧间隔：根据视频实际 fps 与希望处理的每秒帧数 self.process_fps
        if fps and self.process_fps and self.process_fps > 0:
            process_interval = max(1, int(round(float(fps) / float(self.process_fps))))
        else:
            process_interval = 1

        print(f"抽帧设置: 每 {process_interval} 帧处理一次 (目标处理 {self.process_fps} 帧/s)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 决定当前帧是否为处理帧（抽帧）
            process_this_frame = ((frame_count - 1) % process_interval) == 0

            alarms = []
            resized_alarm_img = None
            alarm_image_base64 = None

            if process_this_frame:
                # 应用ROI mask（将所有ROI外区域变黑）
                masked_frame = self.roi_manager.apply_mask(frame)

                # 在masked frame上进行YOLO检测
                detections = self.detector.detect(
                    masked_frame,
                    conf_thresh=0.25,
                    iou_thresh=0.7,
                    target_size=self.target_size
                )

                # 更新入侵状态并触发报警
                alarms = self.alarm_manager.update_intrusion(detections, frame)
                total_alarms += len(alarms)

                # 可视化包含检测框与报警信息
                vis_frame = self._visualize(frame, detections, alarms)

                # 为报警添加可视化图片（先缩放到指定大小，再Base64编码）
                if alarms:
                    # 使用 AlarmManager 提供的缩放与编码方法
                    resized_alarm_img, alarm_image_base64 = self.alarm_manager.resize_and_encode(vis_frame)

                    for alarm in alarms:
                        alarm['image'] = alarm_image_base64

                        # 发送报警到接口
                        if self.alarm_manager.alarm_url:
                            self.alarm_manager._send_alarm(alarm)
            else:
                # 非处理帧：仅绘制 ROI（轻量），不运行检测/报警逻辑
                vis_frame = self.roi_manager.draw_rois(frame.copy(), color=(0, 255, 0), thickness=2)

            # 写入输出视频
            if writer:
                writer.write(vis_frame)

            # 显示
            if display:
                cv2.imshow('Intrusion Detection', vis_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n用户中断")
                    break
                elif key == ord('s'):
                    screenshot_path = f'screenshot_frame{frame_count}.jpg'
                    cv2.imwrite(screenshot_path, vis_frame)
                    print(f"截图保存: {screenshot_path}")

            # 进度显示
            if frame_count % (fps * 5) == 0:  # 每5秒显示一次
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                if total_frames is not None:
                    # 视频文件：显示进度
                    progress = frame_count / total_frames * 100
                    print(f"进度: {frame_count}/{total_frames} ({progress:.1f}%), "
                          f"FPS: {fps_actual:.1f}, 累计报警: {total_alarms}")
                else:
                    # 流媒体：只显示帧数
                    print(f"已处理: {frame_count} 帧, "
                          f"FPS: {fps_actual:.1f}, 累计报警: {total_alarms}")

        # 清理
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        # 统计信息
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"{'='*60}")
        print(f"总帧数: {frame_count}")
        print(f"处理时间: {elapsed:.1f}s")
        print(f"平均FPS: {frame_count/elapsed:.1f}")
        print(f"总报警次数: {total_alarms}")
        print(f"{'='*60}\n")

    def _visualize(self, frame: np.ndarray, detections: List[Dict],
                   alarms: List[Dict]) -> np.ndarray:
        """
        可视化检测结果

        Args:
            frame: 原始帧
            detections: 检测结果
            alarms: 当前帧的报警列表

        Returns:
            可视化后的帧
        """
        vis_frame = frame.copy()

        # 绘制ROI区域
        vis_frame = self.roi_manager.draw_rois(vis_frame, color=(0, 255, 0), thickness=2)

        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['conf']
            cls = det['cls']

            # 检查是否达到报警阈值
            is_alarm = conf >= self.alarm_manager.conf_threshold

            # 根据是否报警选择颜色
            color = (0, 0, 255) if is_alarm else (255, 144, 30)  # 红色=报警，橙色=正常

            # 绘制框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # 标签
            cls_name = self.class_names.get(cls, str(cls))
            label = f'{cls_name} {conf:.2f}'

            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_h - 10),
                         (x1 + label_w, y1), color, -1)

            # 标签文字
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示报警信息
        if alarms:
            alarm_text = "ALARM! INTRUSION DETECTED"
            cv2.putText(vis_frame, alarm_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 显示统计信息
        info_y = vis_frame.shape[0] - 60
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        is_intrusion = self.alarm_manager.intrusion_state['first_time'] is not None
        status = "INTRUSION" if is_intrusion else "NORMAL"
        status_color = (0, 0, 255) if is_intrusion else (0, 255, 0)
        cv2.putText(vis_frame, f"Status: {status}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return vis_frame


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='区域入侵检测系统')

    # 模型参数
    parser.add_argument('--model-yaml', type=str,
                       default="ultralytics/cfg/models/11/yolo11x.yaml",
                       help='模型配置YAML文件')
    parser.add_argument('--weights', type=str,
                       default='data/LLVIP_IF-yolo11x-e300-16-pretrained.pt',
                       help='模型权重文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0 或 cpu)')

    # ROI配置
    parser.add_argument('--config', type=str,
                       default='area_intrusion/roi_config.json',
                       help='ROI配置文件')

    # 输入输出
    parser.add_argument('--source', type=str,
                       default='data/dataset/video_IR/INO_ParkingEvening_T.avi',
                       help='视频路径、摄像头ID(0,1,...)或RTSP地址')
    parser.add_argument('--output-dir', type=str,
                       default='runs/intrusion_detection',
                       help='输出目录')
    parser.add_argument('--save', action='store_true',
                       help='保存输出视频')
    parser.add_argument('--show', action='store_true',
                       help='显示实时结果')
    parser.add_argument('--save-alarms', action='store_true', default=True,
                       help='保存报警截图')

    # 报警参数
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--first-alarm-duration', type=float, default=1.0,
                       help='首次报警时间（秒）- 消抖')
    parser.add_argument('--repeat-alarm-interval', type=float, default=30.0,
                       help='重复报警间隔（秒）')
    parser.add_argument('--tolerance-time', type=float, default=3.0,
                       help='容忍时间（秒）- 检测不到目标后的宽限期')
    parser.add_argument('--save-width', type=int, default=640,
                       help='保存的报警图片宽度')
    parser.add_argument('--save-height', type=int, default=480,
                       help='保存的报警图片高度')
    parser.add_argument('--target-size', type=int, default=640,
                       help='YOLO 检测输入/目标尺寸 (target_size)')
    parser.add_argument('--process-fps', type=float, default=5.0,
                       help='每秒处理帧数（抽帧），例如 2 表示每秒抽取2帧进行检测）')
    parser.add_argument('--alarm-url', type=str, default=None,
                       help='报警接口URL')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("区域入侵检测系统")
    print("=" * 60)

    # 1. 初始化检测器
    print("\n[1/3] 初始化YOLO检测器...")
    detector = Detector(args.model_yaml, args.weights, args.device)

    # 2. 初始化入侵检测系统
    print("\n[2/3] 初始化入侵检测系统...")
    ids = IntrusionDetectionSystem(
        detector=detector,
        roi_config_path=args.config,
        conf_threshold=args.conf_threshold,
        first_alarm_duration=args.first_alarm_duration,
        repeat_alarm_interval=args.repeat_alarm_interval,
        tolerance_time=args.tolerance_time,
        save_height=args.save_height,
        save_width=args.save_width,
        target_size=args.target_size,
        process_fps=args.process_fps,
        alarm_url=args.alarm_url
    )

    # 3. 处理输入
    print("\n[3/3] 开始处理...")

    source = args.source
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定输出路径
    output_path = None
    if args.save:
        if Path(source).exists() and Path(source).is_file():
            # 视频文件
            output_filename = f"{Path(source).stem}_intrusion.mp4"
        else:
            # 流媒体或摄像头
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"stream_{timestamp}.mp4"
        output_path = str(output_dir / output_filename)

    # 判断输入类型并处理
    if Path(source).exists() and Path(source).is_file():
        # 视频文件
        ids.process_video(
            video_path=source,
            output_path=output_path,
            display=args.show,
            save_alarms=args.save_alarms
        )
    else:
        # 摄像头ID或RTSP流
        ids.run_camera(
            source=source,
            output_path=output_path,
            display=args.show,
            save_alarms=args.save_alarms
        )

    print("\n✓ 所有任务完成!")


if __name__ == '__main__':
    main()
