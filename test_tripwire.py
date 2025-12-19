"""
测试新的绊线入侵规则 - Test Tripwire Intrusion Rule

功能：
- 测试 unified_detector 框架中的 TripwireRule
- 使用本地绊线配置文件
- 不依赖后端API，纯本地测试
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import json
import time
import logging
import base64
import numpy as np
from pathlib import Path
from unified_detector.core.detector import UnifiedDetector
from unified_detector.rules.tripwire import TripwireRule
from unified_detector.utils.geometry import draw_tripwires, draw_detections, draw_alarm_text
import warnings
warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)


def load_tripwire_config(config_path: str):
    """加载绊线配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"✓ 加载绊线配置: {len(config['tripwires'])} 条绊线")
    for tw in config['tripwires']:
        print(f"  - {tw['id']}: {tw['direction']}, cooldown={tw['alert_cooldown']}s")

    return config


def convert_tripwires_to_actual_size(tripwires, config_width, config_height, actual_width, actual_height):
    """将绊线从配置尺寸转换到实际视频尺寸"""
    scale_x = actual_width / config_width
    scale_y = actual_height / config_height

    converted_lines = []
    for tw in tripwires:
        if not tw.get('enabled', True):
            continue
        converted_points = []
        for x, y in tw['points']:
            converted_points.append([int(x * scale_x), int(y * scale_y)])
        converted_lines.append(converted_points)

    print(f"✓ 绊线坐标转换: {config_width}x{config_height} → {actual_width}x{actual_height}")
    return converted_lines


def create_rule_config(tripwire_lines, actual_width, actual_height, direction='bidirectional', repeated_alarm_time=10.0):
    """创建规则配置（模拟API返回的配置格式）"""
    rule_config = {
        'enabled': True,
        'sensitivity': 0.45,  # 对应前端sensitivity=5
        'repeated_alarm_time': repeated_alarm_time,  # 测试用，10秒重复报警间隔
        'direction': direction,  # 'left_to_right', 'right_to_left', 'bidirectional'
        'frontend_width': actual_width,
        'frontend_height': actual_height,
        'tripwire_arrays': tripwire_lines,  # 转换后的绊线坐标
        'device_info': {
            'device_id': 'test_device',
            'device_name': '测试摄像头',
            'channel_id': 'test_channel',
            'channel_name': '测试通道'
        }
    }
    return rule_config


def main():
    """主函数"""
    print("=" * 60)
    print("绊线入侵规则测试 - Tripwire Intrusion Rule Test")
    print("=" * 60)

    # 1. 配置参数
    tripwire_config_path = "tripwire_intrusion/config_line.json"
    video_source = r"data\dataset\video_IR\test3.mp4"  # 0=摄像头, 或者视频文件路径

    # 假设配置文件中的坐标是基于640x480的
    config_width = 640
    config_height = 480

    model_yaml = "ultralytics/cfg/models/11/yolo11x.yaml"
    model_weights = "data/LLVIP_IF-yolo11x-e300-16-pretrained.pt"
    device = "cuda:0"  # 或 "cpu"
    tracker = "bytetrack"
    target_size = 800
    conf_threshold = 0.25

    # 2. 加载绊线配置
    print("\n[1/6] 加载绊线配置...")
    tripwire_config = load_tripwire_config(tripwire_config_path)

    # 3. 打开视频流
    print(f"\n[2/6] 打开视频流...")
    if isinstance(video_source, int):
        print(f"  使用摄像头: {video_source}")
    else:
        print(f"  使用视频文件: {video_source}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("✗ 无法打开视频流")
        return

    # 获取视频流信息
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 120:
        fps = 30

    print(f"✓ 视频流信息: {actual_width}x{actual_height} @ {fps}fps")

    # 4. 转换绊线坐标
    print(f"\n[3/6] 转换绊线坐标...")
    converted_lines = convert_tripwires_to_actual_size(
        tripwire_config['tripwires'],
        config_width,
        config_height,
        actual_width,
        actual_height
    )

    # 5. 初始化检测器
    print(f"\n[4/6] 初始化YOLO检测器...")
    detector = UnifiedDetector(model_yaml, model_weights, device, tracker)
    print("✓ 检测器初始化完成")

    # 6. 初始化规则引擎
    print(f"\n[5/6] 初始化绊线入侵规则...")
    rule_config = create_rule_config(
        converted_lines,
        actual_width,
        actual_height,
        direction='bidirectional',  # 可以改成 'left_to_right' 或 'right_to_left'
        repeated_alarm_time=1.0
    )
    rule = TripwireRule(rule_config, camera_key="test_camera")

    # 设置图像高度（用于坐标系转换）
    rule.monitor.set_image_height(actual_height)

    print("✓ 规则引擎初始化完成")
    print(f"  - Sensitivity: {rule.sensitivity}")
    print(f"  - Direction: {rule.direction}")
    print(f"  - Repeated alarm time: {rule.repeated_alarm_time}s")
    print(f"  - Tripwire lines: {len(converted_lines)}")

    # 7. 主循环
    print(f"\n[6/6] 开始处理循环...")
    print("按 'q' 退出, 按 's' 截图, 按 'r' 重置规则状态\n")

    frame_count = 0
    process_interval = 5 
    times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ 读帧失败")
                break

            frame_count += 1
            current_time = time.time()

            # 每隔process_interval帧处理一次
            if (frame_count - 1) % process_interval == 0:
                # 检测和跟踪（必须开启跟踪）
                time_1 = time.time()
                detections = detector.detect_and_track(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=0.7,
                    target_size=target_size
                )

                # 规则处理
                alarm_info = rule.process(frame, detections, current_time)
                time2 = time.time()
                times.append((time2 - time_1)* 1000)

                # 可视化
                vis_frame = frame.copy()

                # 绘制绊线
                vis_frame = draw_tripwires(vis_frame, converted_lines, color=(0, 255, 255), thickness=3)

                # 绘制检测框和轨迹
                vis_frame = draw_detections(vis_frame, detections, conf_threshold=rule.sensitivity,
                                           class_names={0: 'person'})

                # 绘制轨迹
                for track_id, track in rule.track_history.items():
                    # 绘制轨迹点
                    trajectory = list(track.trajectory)
                    if len(trajectory) > 1:
                        for i in range(len(trajectory) - 1):
                            pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
                            pt2 = (int(trajectory[i+1][0]), int(trajectory[i+1][1]))
                            cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)

                # 显示状态信息
                status_text = f"Frame: {frame_count} | Detections: {len(detections)} | Tracks: {len(rule.track_history)}"
                cv2.putText(vis_frame, status_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 如果有报警，显示报警信息
                if alarm_info:
                    vis_frame = draw_alarm_text(vis_frame, "TRIPWIRE CROSSED!")
                    # 手动更新报警时间（测试中没有真实API）
                    rule.last_alarm_time = current_time
                    base64_image = alarm_info.get('alarmPicture', '')
                    #保存报警截图
                    image_data = cv2.imdecode(
                        np.frombuffer(base64.b64decode(base64_image), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    alarm_image_path = f"runs/tripwire_new/alarm_{frame_count}.jpg"
                    cv2.imwrite(alarm_image_path, image_data)

                # 显示图像
                cv2.imshow('Tripwire Test', vis_frame)

            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('s'):
                # 截图
                screenshot_path = f"screenshot_tripwire_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, vis_frame)
                print(f"✓ 截图保存: {screenshot_path}")
            elif key == ord('r'):
                # 重置规则状态
                rule.reset()
                print("✓ 规则状态已重置")
        print(sum(times)/len(times) if times else 0)
    except KeyboardInterrupt:
        print("\n\n用户中断")

    finally:
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ 测试结束")


if __name__ == '__main__':
    main()
