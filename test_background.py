import cv2
import numpy as np
import time
import os
from datetime import datetime


def calculate_iou(box1, box2):
    """
    计算两个边界框的IOU (Intersection over Union)

    Args:
        box1: (x, y, w, h)
        box2: (x, y, w, h)

    Returns:
        iou: 交并比值
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集区域
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算各自面积
    box1_area = w1 * h1
    box2_area = w2 * h2

    # 计算并集
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def calculate_distance(box1, box2):
    """
    计算两个边界框中心点之间的欧氏距离

    Args:
        box1: (x, y, w, h)
        box2: (x, y, w, h)

    Returns:
        distance: 中心点距离
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算中心点
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2

    # 欧氏距离
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


def merge_rois(boxes, iou_threshold=0.3, distance_threshold=100):
    """
    基于IOU和距离合并ROI

    Args:
        boxes: ROI列表，每个元素为 (x, y, w, h)
        iou_threshold: IOU阈值，超过此值认为重叠
        distance_threshold: 距离阈值，小于此值认为邻近

    Returns:
        merged_boxes: 合并后的ROI列表
    """
    if len(boxes) == 0:
        return []

    # 标记是否已合并
    merged = [False] * len(boxes)
    merged_boxes = []

    for i in range(len(boxes)):
        if merged[i]:
            continue

        # 当前box及其待合并的boxes
        current_group = [boxes[i]]
        merged[i] = True

        for j in range(i + 1, len(boxes)):
            if merged[j]:
                continue

            # 计算IOU和距离
            iou = calculate_iou(boxes[i], boxes[j])
            distance = calculate_distance(boxes[i], boxes[j])

            # 如果IOU超过阈值或距离小于阈值，则合并
            if iou > iou_threshold or distance < distance_threshold:
                current_group.append(boxes[j])
                merged[j] = True

        # 合并当前组的所有boxes
        if len(current_group) > 0:
            x_min = min([box[0] for box in current_group])
            y_min = min([box[1] for box in current_group])
            x_max = max([box[0] + box[2] for box in current_group])
            y_max = max([box[1] + box[3] for box in current_group])

            merged_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            merged_boxes.append(merged_box)

    return merged_boxes


def save_roi_images(frame, boxes, output_dir, frame_number):
    """
    保存每个ROI区域的图片

    Args:
        frame: 原始帧
        boxes: ROI列表
        output_dir: 输出目录
        frame_number: 帧编号
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        # 确保坐标在图像范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w > 0 and h > 0:
            roi = frame[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{frame_number:06d}_roi_{idx:02d}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, roi)


def mog2_gpu_accelerated_demo(video_path=None, use_camera=True, frame_skip=0,
                              iou_threshold=0.3, distance_threshold=100,
                              save_rois=False, output_dir="roi_output",
                              resize_width=800, resize_height=800):
    """
    使用GPU加速的MOG2背景减除算法

    Args:
        video_path: 视频文件路径
        use_camera: 是否使用摄像头
        frame_skip: 抽帧间隔，0表示不抽帧，1表示每隔1帧处理一次，以此类推
        iou_threshold: IOU阈值，用于合并ROI
        distance_threshold: 距离阈值，用于合并ROI
        save_rois: 是否保存ROI图片
        output_dir: ROI图片输出目录
        resize_width: 压缩后的宽度
        resize_height: 压缩后的高度
    """
    # 检查CUDA可用性
    cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if not cuda_enabled:
        print("警告: 未检测到可用的CUDA设备，将回退到CPU版本")

    if cuda_enabled:
        print(f"使用GPU加速 (CUDA设备数: {cv2.cuda.getCudaEnabledDeviceCount()})")

        # 使用CUDA版本的MOG2背景减除器
        # 注意: cuda.createBackgroundSubtractorMOG2 是CUDA特定的
        try:
            # 初始化CUDA流
            stream = cv2.cuda_Stream()

            # 创建CUDA MOG2背景减除器
            # 参数说明:
            # - history: 学习率，值越大背景更新越慢
            # - varThreshold: 方差阈值
            # - detectShadows: 是否检测阴影
            mog2_cuda = cv2.cuda.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )

            use_gpu = True
            print("使用CUDA加速的MOG2")
        except:
            print("CUDA MOG2不可用，回退到CPU版本")
            mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            use_gpu = False
    else:
        mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        use_gpu = False

    # 打开视频或摄像头
    if video_path:
        cap = cv2.VideoCapture(video_path)
    elif use_camera:
        cap = cv2.VideoCapture(0)
    else:
        print("请提供视频路径或使用摄像头")
        return

    if not cap.isOpened():
        print("无法打开视频源")
        return

    print("按 'q' 退出")
    print("按 's' 保存当前帧")
    print("按 'g' 切换GPU/CPU模式")
    if frame_skip > 0:
        print(f"抽帧模式: 每隔 {frame_skip} 帧处理一次")

    frame_count = 0
    processed_count = 0
    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束")
            break

        # 压缩图片到指定尺寸
        frame = cv2.resize(frame, (resize_width, resize_height))

        frame_count += 1

        # 抽帧逻辑：跳过指定数量的帧
        if frame_skip > 0 and (frame_count - 1) % (frame_skip + 1) != 0:
            continue

        processed_count += 1

        # 计算FPS
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = processed_count
            processed_count = 0
            prev_time = current_time

        if use_gpu:
            # GPU处理流程
            # 将帧上传到GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # 应用MOG2
            gpu_fg_mask = mog2_cuda.apply(gpu_frame, learningRate=-1, stream=stream)

            # 下载回CPU
            fg_mask = gpu_fg_mask.download()

            # CUDA流同步
            stream.waitForCompletion()
        else:
            # CPU处理流程
            fg_mask = mog2.apply(frame)

        # 后续处理（阈值化、形态学操作等）
        fg_mask_no_shadow = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask_clean = cv2.morphologyEx(fg_mask_no_shadow, cv2.MORPH_OPEN, kernel)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel)

        # 提取轮廓并转换为边界框
        contours, _ = cv2.findContours(fg_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 收集所有ROI
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))

        # 合并ROI
        merged_boxes = merge_rois(boxes, iou_threshold=iou_threshold, distance_threshold=distance_threshold)

        # 保存ROI图片
        if save_rois and len(merged_boxes) > 0:
            save_roi_images(frame, merged_boxes, output_dir, frame_count)

        # 在原图上绘制合并后的边界框
        result_frame = frame.copy()
        for box in merged_boxes:
            x, y, w, h = box
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 显示ROI编号
            cv2.putText(result_frame, f"ROI", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示模式信息和FPS
        mode_text = f"GPU Accelerated" if use_gpu else "CPU Mode"
        fps_text = f"FPS: {fps}"
        frame_info = f"Frame: {frame_count}"
        roi_info = f"ROIs: {len(merged_boxes)}"
        if frame_skip > 0:
            frame_info += f" (Skip: {frame_skip})"

        cv2.putText(result_frame, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, fps_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, frame_info, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, roi_info, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('GPU MOG2 Motion Detection', result_frame)
        cv2.imshow('Foreground Mask', fg_mask_clean)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'gpu_mog2_result_{int(time.time())}.jpg', result_frame)
            print("已保存结果")
        elif key == ord('g') and cuda_enabled:
            use_gpu = not use_gpu
            print(f"切换到 {'GPU' if use_gpu else 'CPU'} 模式")

    cap.release()
    cv2.destroyAllWindows()

def check_cuda_capabilities():
    """检查CUDA和OpenCV的GPU支持情况"""
    print("="*50)
    print("CUDA/GPU支持检查")
    print("="*50)

    # 检查CUDA设备
    cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA设备数量: {cuda_device_count}")

    if cuda_device_count > 0:
        for i in range(cuda_device_count):
            device_info = cv2.cuda.getDevice(i)
            print(f"设备 {i}: {device_info.name()}")
            print(f"  Compute Capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
            print(f"  多处理器数量: {device_info.multiProcessorCount()}")
    else:
        print("未检测到CUDA设备")

    # 检查OpenCV构建信息
    print("\nOpenCV构建信息:")
    build_info = cv2.getBuildInformation()

    # 查找关键信息
    info_keys = ["CUDA", "CUDNN", "WITH_CUDA", "WITH_CUDNN", "CUDA_ARCH_BIN"]
    for key in info_keys:
        lines = [line for line in build_info.split('\n') if key in line]
        for line in lines[:3]:  # 只显示前3行相关结果
            print(f"  {line}")

    print("="*50)


if __name__ == "__main__":
    # 检查CUDA支持
    check_cuda_capabilities()

    # 配置参数
    video_path = "data/ls2.mp4"
    frame_skip = 5  # 抽帧间隔，0表示不抽帧
    iou_threshold = 0.01  # IOU阈值
    distance_threshold = 50  # 距离阈值
    save_rois = True  # 是否保存ROI图片
    output_dir = "roi_output"  # ROI输出目录
    resize_width = 2000  # 压缩后的宽度
    resize_height = 1000  # 压缩后的高度

    print(f"\n运行配置:")
    print(f"视频路径: {video_path}")
    print(f"抽帧间隔: {frame_skip}")
    print(f"IOU阈值: {iou_threshold}")
    print(f"距离阈值: {distance_threshold}")
    print(f"保存ROI: {save_rois}")
    print(f"输出目录: {output_dir}")
    print(f"图片尺寸: {resize_width}x{resize_height}")

    mog2_gpu_accelerated_demo(
        video_path=video_path,
        use_camera=False,
        frame_skip=frame_skip,
        iou_threshold=iou_threshold,
        distance_threshold=distance_threshold,
        save_rois=save_rois,
        output_dir=output_dir,
        resize_width=resize_width,
        resize_height=resize_height
    )