import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector import Detector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VideoDetector:
    """视频检测器 - 支持抽帧检测"""
    
    def __init__(self, 
                 yaml_path: str, 
                 weights_path: str, 
                 device: str = 'cuda:0',
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.7,
                 target_size: int = 640,
                 fps_target: int = 5):  # 新增：目标FPS
        """
        初始化检测器
        
        Args:
            yaml_path: 模型YAML文件路径
            weights_path: 权重文件路径
            device: 设备 (cuda:0 或 cpu)
            conf_thresh: 置信度阈值
            iou_thresh: IOU阈值
            target_size: 检测尺寸
            fps_target: 目标处理帧率（每秒检测帧数）
        """
        self.yaml_path = yaml_path
        self.weights_path = weights_path
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_size = target_size
        self.fps_target = fps_target  # 目标每秒检测5帧
        
        # 类别名称（根据您的模型调整）
        self.class_names = {0: 'person'}
        
        # 初始化检测器
        logger.info("初始化检测器...")
        logger.info(f"模型YAML: {yaml_path}")
        logger.info(f"权重文件: {weights_path}")
        logger.info(f"设备: {device}")
        logger.info(f"目标检测帧率: {fps_target} FPS")
        
        self.detector = Detector(
            yaml_path=yaml_path,
            weights_path=weights_path,
            device=device
        )
        
        # 性能统计
        self.frame_count = 0
        self.total_fps = []
        self.total_inference_time = []
        
        # 抽帧相关
        self.last_detections = []  # 上一帧的检测结果
        self.last_detection_time = 0  # 上一次检测的时间
        
        # 报警相关
        self.alarm_history = []
        self.alarm_count = 0
        
        # 创建输出目录
        self.output_dir = Path("runs/video_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("检测器初始化完成")
    
    def process_video_file(self, 
                          video_path: str,
                          output_path: Optional[str] = None,
                          max_frames: int = 0,
                          show_display: bool = True,
                          save_video: bool = True,
                          save_json: bool = True):
        """
        处理视频文件 - 每秒抽5帧检测
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径，None则自动生成
            max_frames: 最大处理帧数，0表示全部
            show_display: 是否显示实时画面
            save_video: 是否保存输出视频
            save_json: 是否保存检测结果JSON
        """
        logger.info(f"开始处理视频文件: {video_path}")
        logger.info(f"抽帧策略: 每秒检测 {self.fps_target} 帧")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频信息: {width}x{height} @ {video_fps}fps, 总帧数: {total_frames}")
        
        # 计算抽帧间隔
        if video_fps <= 0:
            logger.warning("无法获取视频FPS，使用默认值30")
            video_fps = 30
        
        # 每多少帧检测一次 (每秒检测5帧)
        detection_interval = max(1, int(video_fps / self.fps_target))
        logger.info(f"抽帧间隔: 每 {detection_interval} 帧检测一次")
        
        # 自动生成输出路径
        if output_path is None:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"{video_name}_detection_{self.fps_target}fps_{timestamp}.mp4")
        
        # 创建视频写入器
        video_writer = None
        if save_video:
            # 使用原视频的FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                video_fps,  # 保持原视频FPS
                (width, height)
            )
            logger.info(f"输出视频: {output_path}")
        
        # 创建JSON结果文件
        json_results = []
        json_path = output_path.replace('.mp4', '.json') if save_json else None
        
        # 性能统计
        total_time = 0
        processed_frames = 0
        frame_idx = 0
        
        # 主循环
        logger.info("开始处理视频...")
        print("=" * 60)
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 限制最大帧数
            if max_frames > 0 and frame_idx >= max_frames:
                break
            
            # 判断是否需要检测当前帧
            should_detect = (frame_idx % detection_interval == 1)  # 每秒的第1帧检测
            
            if should_detect:
                # 计时开始
                inference_start = time.time()
                
                # 检测
                detections = self.detector.detect(
                    frame,
                    conf_thresh=self.conf_thresh,
                    iou_thresh=self.iou_thresh,
                    target_size=self.target_size
                )
                
                # 计时结束
                inference_time = time.time() - inference_start
                self.total_inference_time.append(inference_time)
                
                # 保存检测结果
                self.last_detections = detections
                self.last_detection_time = time.time()
                
                # 统计
                processed_frames += 1
                self.frame_count += 1
                total_time += inference_time
                
                # 计算实时FPS
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                self.total_fps.append(current_fps)
                
                # 报警检测
                if len(detections) > 0:
                    self._check_alarm(detections, frame_idx)
                
                # 保存结果到JSON
                if save_json:
                    frame_result = {
                        'frame_index': frame_idx,
                        'timestamp': time.time(),
                        'detection_count': len(detections),
                        'inference_time_ms': inference_time * 1000,
                        'fps': current_fps,
                        'detections': [
                            {
                                'box': det['box'],
                                'confidence': float(det['conf']),
                                'class_id': int(det['cls']),
                                'class_name': self.class_names.get(int(det['cls']), 'unknown')
                            }
                            for det in detections
                        ]
                    }
                    json_results.append(frame_result)
            else:
                # 使用上一次的检测结果
                detections = self.last_detections
                inference_time = 0
                current_fps = 0
            
            # 可视化（每一帧都绘制，但检测结果可能来自前一秒）
            vis_frame = self._visualize_detection(
                frame, 
                detections, 
                current_fps, 
                inference_time,
                frame_idx,
                should_detect
            )
            
            # 保存视频帧
            if video_writer:
                video_writer.write(vis_frame)
            
            # 显示
            if show_display:
                cv2.imshow('Video Detection (5 FPS)', vis_frame)
                
                # 按ESC或Q退出
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:
                    logger.info("用户中断处理")
                    break
            
            # 打印进度
            if frame_idx % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                progress = (frame_idx / min(total_frames, max_frames)) * 100 if max_frames > 0 else (frame_idx / total_frames) * 100
                
                # 计算实际检测帧率
                detection_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\r进度: {progress:.1f}% | 总帧: {frame_idx}/{total_frames} | "
                      f"已检测: {processed_frames}帧 | 显示FPS: {avg_fps:.1f} | "
                      f"检测FPS: {detection_fps:.1f}/{self.fps_target} | "
                      f"检测数: {len(detections)}", end='', flush=True)
        
        # 清理
        cap.release()
        if video_writer:
            video_writer.release()
        
        if show_display:
            cv2.destroyAllWindows()
        
        # 保存JSON结果
        if save_json and json_results:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"检测结果保存到: {json_path}")
        
        # 打印统计信息
        self._print_statistics(total_time, processed_frames, frame_idx)
        
        return json_path if save_json else None
    
    def _visualize_detection(self, 
                           frame: np.ndarray, 
                           detections: List[Dict], 
                           fps: float, 
                           inference_time: float,
                           frame_index: int,
                           is_detection_frame: bool) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            frame: 原始帧
            detections: 检测结果
            fps: 当前FPS
            inference_time: 推理时间
            frame_index: 帧索引
            is_detection_frame: 是否是检测帧
        
        Returns:
            可视化后的帧
        """
        vis_frame = frame.copy()
        
        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['conf']
            cls = det['cls']
            
            # 颜色
            color = self._get_color(cls)
            
            # 绘制矩形
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
    
        
        return vis_frame
    
    def _check_alarm(self, detections: List[Dict], frame_index: int):
        """
        检查是否触发报警
        
        Args:
            detections: 检测结果
            frame_index: 帧索引
        """
        # 简单的报警逻辑：检测到目标就报警
        if len(detections) > 0:
            alarm = {
                'timestamp': time.time(),
                'frame_index': frame_index,
                'detection_count': len(detections),
                'details': [
                    {
                        'class_id': int(det['cls']),
                        'confidence': float(det['conf']),
                        'box': det['box']
                    }
                    for det in detections
                ]
            }
            self.alarm_history.append(alarm)
            self.alarm_count += 1
            
            # 每5次报警记录一次
            if self.alarm_count % 5 == 0:
                logger.info(f"报警触发! 帧{frame_index}检测到 {len(detections)} 个目标")
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """根据类别生成颜色"""
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]
        return colors[class_id % len(colors)]
    
    def _print_statistics(self, total_time: float, detection_frames: int, total_frames: int):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("处理统计")
        print("=" * 60)
        
        if detection_frames > 0:
            elapsed_time = total_frames / 30 if total_frames > 0 else 0  # 假设30fps
            avg_detection_fps = detection_frames / elapsed_time if elapsed_time > 0 else 0
            avg_inference = (total_time / detection_frames * 1000) if detection_frames > 0 else 0
            
            if self.total_fps:
                max_fps = max(self.total_fps)
                min_fps = min(self.total_fps)
            else:
                max_fps = min_fps = 0
            
            print(f"视频总帧数: {total_frames}")
            print(f"实际检测帧数: {detection_frames}")
            print(f"检测帧比例: {detection_frames/total_frames*100:.1f}%")
            print(f"总检测时间: {total_time:.2f}s")
            print(f"平均检测FPS: {avg_detection_fps:.2f} (目标: {self.fps_target} FPS)")
            print(f"最大检测FPS: {max_fps:.2f}")
            print(f"最小检测FPS: {min_fps:.2f}")
            print(f"平均推理时间: {avg_inference:.2f}ms")
            print(f"报警次数: {self.alarm_count}")
            print(f"输出目录: {self.output_dir.absolute()}")
        
        print("=" * 60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='落水检测系统 - 视频测试脚本(抽帧版)')
    
    # 模型参数
    parser.add_argument('--yaml', type=str, 
                       default='ultralytics/cfg/models/11/yolo11n.yaml',
                       help='模型YAML配置文件路径')
    parser.add_argument('--weights', type=str,
                       default='data/LLVIP_IF-yolo11n-e300-16-pretrained-.pt',
                       help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0 或 cpu)')
    
    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IOU阈值')
    parser.add_argument('--size', type=int, default=736,
                       help='检测尺寸')
    parser.add_argument('--fps-target', type=int, default=5,
                       help='目标检测帧率(每秒检测帧数)')
    
    # 输入源参数
    parser.add_argument('--input', type=str, default=r'data\dataset\video_IR\test3.mp4',
                       help='输入源: 视频文件路径')
    
    # 处理参数
    parser.add_argument('--max-frames', type=int, default=0,
                       help='最大处理帧数 (0表示无限制)')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='',
                       help='输出视频路径 (不指定则自动生成)')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时画面')
    parser.add_argument('--no-video', action='store_true',
                       help='不保存输出视频')
    parser.add_argument('--no-json', action='store_true',
                       help='不保存JSON结果')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print(f"视频测试 - 抽帧模式 (每秒检测 {args.fps_target} 帧)")
    print("=" * 60)
    
    logger.info(f"输入源: {args.input}")
    logger.info(f"目标检测帧率: {args.fps_target} FPS")
    
    # 初始化检测器
    detector = VideoDetector(
        yaml_path=args.yaml,
        weights_path=args.weights,
        device=args.device,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        target_size=args.size,
        fps_target=args.fps_target
    )
    
    try:
        detector.process_video_file(
            video_path=args.input,
            output_path=args.output if args.output else None,
            max_frames=args.max_frames,
            show_display=not args.no_display,
            save_video=not args.no_video,
            save_json=not args.no_json
        )
    
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
    finally:
        print("\n程序结束")


if __name__ == '__main__':
    main()