"""
批量推理性能测试脚本

测试不同batch size下的推理性能：
- Batch=1 (当前)
- Batch=2
- Batch=4
- Batch=8

对比：
- 总推理时间
- 单张平均时间
- 加速比
- GPU显存占用
"""

import time
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from detector import Detector


def create_test_images(num_images=100, size=(640, 640)):
    """创建测试图片"""
    print(f"生成 {num_images} 张测试图片 (size={size})...")
    images = []
    for i in range(num_images):
        # 生成随机图片
        img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        images.append(img)
    print(f"✓ 测试图片已生成")
    return images


def test_single_inference(model, images, device, conf=0.25, iou=0.7, imgsz=640, warmup=10):
    """测试单张推理 (Batch=1)"""
    print(f"\n{'='*60}")
    print(f"测试: Batch=1 (单张推理)")
    print(f"{'='*60}")

    # 预热
    print(f"预热中 ({warmup}次)...")
    for i in range(warmup):
        _ = model.detect(images[0], conf, iou, imgsz)

    # 正式测试
    print(f"正式测试 ({len(images)}次)...")
    times = []

    for img in images:
        start = time.time()
        results = model.detect(img, conf, iou, imgsz)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"结果:")
    print(f"  平均时间: {avg_time:.2f}ms/张")
    print(f"  标准差:   {std_time:.2f}ms")
    print(f"  最小/最大: {min_time:.2f}ms / {max_time:.2f}ms")

    return avg_time, times


def test_batch_inference(model, images, batch_size, conf=0.25, iou=0.7, imgsz=640, warmup=5):
    """测试批量推理（真正的batch推理）"""
    print(f"\n{'='*60}")
    print(f"测试: Batch={batch_size} (真正的批量推理)")
    print(f"{'='*60}")

    # 将图片分组
    batches = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        if len(batch) == batch_size:  # 只保留完整batch
            batches.append(batch)

    num_batches = len(batches)
    num_images = num_batches * batch_size
    print(f"共 {num_batches} 个batch (总计 {num_images} 张图片)")

    # 预热
    print(f"预热中 ({warmup}次batch)...")
    for i in range(min(warmup, num_batches)):
        _ = model.detect_batch(batches[i], conf, iou, imgsz)

    # 正式测试
    print(f"正式测试 ({num_batches}个batch)...")
    batch_times = []

    for batch in batches:
        start = time.time()
        _ = model.detect_batch(batch, conf, iou, imgsz)
        elapsed = (time.time() - start) * 1000  # ms
        batch_times.append(elapsed)

    avg_batch_time = np.mean(batch_times)
    avg_per_image = avg_batch_time / batch_size
    total_time = sum(batch_times)

    print(f"结果:")
    print(f"  平均batch时间: {avg_batch_time:.2f}ms (batch={batch_size})")
    print(f"  单张平均时间:  {avg_per_image:.2f}ms")
    print(f"  总耗时:        {total_time:.0f}ms ({num_images}张)")

    return avg_per_image, batch_times


def check_gpu_memory():
    """检查GPU显存占用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"  GPU显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
        return allocated, reserved
    return 0, 0


def main():
    # ========== 配置 ==========
    MODEL_YAML = "ultralytics/cfg/models/11/yolo11m.yaml"  
    MODEL_WEIGHTS = "data/LLVIP-yolo11m-e300-16-pretrained.pt"  
    DEVICE = "cuda:0"  # 或 "cuda:1"
    IMG_SIZE = 640
    NUM_TEST_IMAGES = 100
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7
    BATCH_SIZES = [1, 2, 4, 8]  # 要测试的batch sizes

    print("="*60)
    print("YOLO 批量推理性能测试")
    print("="*60)
    print(f"配置:")
    print(f"  模型配置: {MODEL_YAML}")
    print(f"  模型权重: {MODEL_WEIGHTS}")
    print(f"  设备:     {DEVICE}")
    print(f"  图片尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  测试数量: {NUM_TEST_IMAGES}张")
    print(f"  Batch:    {BATCH_SIZES}")
    print("="*60)

    # 检查文件是否存在
    if not Path(MODEL_WEIGHTS).exists():
        print(f"\n❌ 错误: 模型权重文件不存在: {MODEL_WEIGHTS}")
        print("请修改脚本中的 MODEL_WEIGHTS 路径")
        return

    # 加载模型
    print(f"\n正在加载YOLO模型...")
    model = YOLO(MODEL_WEIGHTS)
    print(f"✓ 模型加载成功")

    # 检查初始显存
    print(f"\n初始GPU状态:")
    check_gpu_memory()

    # 生成测试图片
    test_images = create_test_images(NUM_TEST_IMAGES, size=(IMG_SIZE, IMG_SIZE))

    # 存储结果
    results = {}

    detector = Detector(MODEL_YAML, MODEL_WEIGHTS, DEVICE)

    # 测试不同batch size
    for batch_size in BATCH_SIZES:
        if batch_size == 1:
            avg_time, _ = test_single_inference(
                detector, test_images, DEVICE, CONF_THRESHOLD, IOU_THRESHOLD, IMG_SIZE
            )
        else:
            avg_time, _ = test_batch_inference(
                detector, test_images, batch_size, CONF_THRESHOLD, IOU_THRESHOLD, IMG_SIZE
            )

        results[batch_size] = avg_time

        # 显示显存占用
        check_gpu_memory()

    # ========== 汇总结果 ==========
    print(f"\n{'='*60}")
    print(f"性能对比汇总")
    print(f"{'='*60}")

    baseline = results[1]  # Batch=1作为基准

    print(f"{'Batch Size':<12} {'平均时间':<15} {'加速比':<10} {'吞吐量':<15}")
    print(f"{'-'*60}")

    for batch_size in BATCH_SIZES:
        avg_time = results[batch_size]
        speedup = baseline / avg_time
        throughput = 1000 / avg_time  # 帧/秒

        print(f"{batch_size:<12} {avg_time:>8.2f}ms/张    {speedup:>5.2f}x      {throughput:>6.1f} fps")

    print(f"{'='*60}")

    # ========== 实际场景分析 ==========
    print(f"\n实际场景分析 (24路摄像头):")
    print(f"{'-'*60}")

    for batch_size in BATCH_SIZES:
        avg_time = results[batch_size]

        # 计算24路情况下的性能
        time_per_24_frames = 24 * avg_time  # 处理24帧的总时间(ms)
        max_fps_per_camera = 1000 / time_per_24_frames  # 每路最大帧率

        # 如果使用batching，需要加上等待时间
        if batch_size > 1:
            # 假设等待窗口为batch凑齐的时间
            wait_time = (batch_size - 1) * (1000 / 24)  # 粗略估计
            total_latency = avg_time + wait_time
            print(f"Batch={batch_size}: 单张{avg_time:.1f}ms + 等待{wait_time:.0f}ms = {total_latency:.0f}ms延迟, "
                  f"24路最大{max_fps_per_camera:.2f}fps/路")
        else:
            print(f"Batch=1:  单张{avg_time:.1f}ms, 24路最大{max_fps_per_camera:.2f}fps/路")

    print(f"\n建议:")
    # 找到最优的batch size
    best_batch = 1
    best_speedup = 1.0
    for batch_size in BATCH_SIZES:
        if batch_size == 1:
            continue
        speedup = baseline / results[batch_size]
        if speedup > best_speedup:
            best_speedup = speedup
            best_batch = batch_size

    if best_speedup > 1.5:
        print(f"  推荐使用 Batch={best_batch} (单张加速 {best_speedup:.2f}x)")
        print(f"  需要实现动态batching，等待窗口建议 30-50ms")
    else:
        print(f"  批量推理提升有限 ({best_speedup:.2f}x)，可能不值得改造")
        print(f"  建议保持当前的 Batch=1 单张推理")

    print(f"\n✓ 测试完成")


if __name__ == "__main__":
    main()
