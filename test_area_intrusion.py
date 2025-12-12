"""
测试新的区域入侵规则 - Test Area Intrusion Rule

功能：
- 测试 unified_detector 框架中的 AreaIntrusionRule
- 使用本地 ROI 配置文件
- 不依赖后端API，纯本地测试
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import json
import time
import logging
from pathlib import Path
import base64
import numpy as np 
from unified_detector.core.detector import UnifiedDetector
from unified_detector.rules.area_intrusion import AreaIntrusionRule
from unified_detector.utils.geometry import draw_rois, draw_detections, draw_alarm_text
import warnings
warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)


def load_roi_config(config_path: str):
    """加载ROI配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"✓ 加载ROI配置: {len(config['rois'])} 个区域")
    print(f"  配置尺寸: {config['image_width']}x{config['image_height']}")

    return config


def convert_roi_to_actual_size(rois, config_width, config_height, actual_width, actual_height):
    """将ROI从配置尺寸转换到实际视频尺寸"""
    scale_x = actual_width / config_width
    scale_y = actual_height / config_height

    converted_rois = []
    for roi in rois:
        converted_roi = []
        for x, y in roi:
            converted_roi.append([int(x * scale_x), int(y * scale_y)])
        converted_rois.append(converted_roi)

    print(f"✓ ROI坐标转换: {config_width}x{config_height} → {actual_width}x{actual_height}")
    return converted_rois


def create_rule_config(rois, actual_width, actual_height):
    """创建规则配置（模拟API返回的配置格式）"""
    rule_config = {
        'enabled': True,
        'sensitivity': 0.45,  # 对应前端sensitivity=5
        'first_alarm_time': 1.0,  # 首次报警时间1秒
        'repeated_alarm_time': 5.0,  # 测试用，10秒重复报警间隔
        'frontend_width': actual_width,  # 前端显示尺寸
        'frontend_height': actual_height,
        'roi_arrays': rois,  # 转换后的ROI坐标
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
    print("区域入侵规则测试 - Area Intrusion Rule Test")
    print("=" * 60)

    # 1. 配置参数
    roi_config_path = "area_intrusion/roi_config.json"
    video_source = r"data\dataset\video_IR\test12.mp4"  # 0=摄像头, 或者视频文件路径

    model_yaml = "ultralytics/cfg/models/11/yolo11x.yaml"
    model_weights = "data/LLVIP_IF-yolo11x-e300-16-pretrained.pt"
    device = "cuda:0"  # 或 "cpu"
    tracker = "bytetrack"
    target_size = 800
    conf_threshold = 0.25

    # 2. 加载ROI配置
    print("\n[1/6] 加载ROI配置...")
    roi_config = load_roi_config(roi_config_path)

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

    # 4. 转换ROI坐标
    print(f"\n[3/6] 转换ROI坐标...")
    converted_rois = convert_roi_to_actual_size(
        roi_config['rois'],
        roi_config['image_width'],
        roi_config['image_height'],
        actual_width,
        actual_height
    )

    # 5. 初始化检测器
    print(f"\n[4/6] 初始化YOLO检测器...")
    detector = UnifiedDetector(model_yaml, model_weights, device, tracker)
    print("✓ 检测器初始化完成")

    # 6. 初始化规则引擎
    print(f"\n[5/6] 初始化区域入侵规则...")
    rule_config = create_rule_config(converted_rois, actual_width, actual_height)
    rule = AreaIntrusionRule(rule_config, camera_key="test_camera")
    print("✓ 规则引擎初始化完成")
    print(f"  - Sensitivity: {rule.sensitivity}")
    print(f"  - First alarm time: {rule.first_alarm_time}s")
    print(f"  - Tolerance time: {rule.tolerance_time}s")
    print(f"  - Repeated alarm time: {rule.repeated_alarm_time}s")

    # 7. 主循环
    print(f"\n[6/6] 开始处理循环...")
    print("按 'q' 退出, 按 's' 截图, 按 'r' 重置规则状态\n")

    frame_count = 0
    process_interval = 5  # 每隔多少帧处理一次
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
                # 检测和跟踪
                detect_start_time = time.time()
                detections = detector.detect_and_track(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=0.7,
                    target_size=target_size
                )

                # 规则处理
                alarm_info = rule.process(frame, detections, current_time)
                detect_elapsed_ms = (time.time() - detect_start_time) * 1000
                times.append(detect_elapsed_ms)

                # 可视化
                vis_frame = frame.copy()

                # 绘制ROI
                vis_frame = draw_rois(vis_frame, converted_rois, color=(0, 255, 0), thickness=2)

                # 绘制检测框
                vis_frame = draw_detections(vis_frame, detections, conf_threshold=rule.sensitivity,
                                           class_names={0: 'person'})

                # 显示入侵状态
                if rule.intrusion_state['first_time'] is not None:
                    duration = current_time - rule.intrusion_state['first_time']
                    state_text = f"INTRUSION! Duration: {duration:.1f}s"
                    cv2.putText(vis_frame, state_text, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 如果有报警，显示报警信息
                if alarm_info:
                    vis_frame = draw_alarm_text(vis_frame, "ALARM TRIGGERED!")

                    # 手动更新报警时间（测试中没有真实API）
                    rule.last_alarm_time = current_time
                    base64_image = alarm_info.get('alarmPicCode', '')
                    #保存报警截图
                    image_data = cv2.imdecode(
                        np.frombuffer(base64.b64decode(base64_image), np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    alarm_image_path = f"runs/area_intrusion_new/alarm_{frame_count}.jpg"
                    cv2.imwrite(alarm_image_path, image_data)

                # 显示图像
                cv2.imshow('Area Intrusion Test', vis_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('s'):
                # 截图
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, vis_frame)
                print(f"✓ 截图保存: {screenshot_path}")
            elif key == ord('r'):
                # 重置规则状态
                rule.reset()
                print("✓ 规则状态已重置")
        print(sum(times) / len(times) if times else 0)

    except KeyboardInterrupt:
        print("\n\n用户中断")

    finally:
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ 测试结束")


if __name__ == '__main__':
    main()
