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

支持模型格式：
- .pt (PyTorch)
- .engine (TensorRT)
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

def test_single_inference(model, images, device, conf=0.25, iou=0.7, imgsz=640, warmup=100):
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

def test_tensorrt_performance(engine_path, images, device, conf=0.25, iou=0.7, imgsz=640, warmup=10, channels=3, use_simotm='RGB'):
    """
    测试TensorRT engine的推理性能

    Args:
        engine_path: .engine文件路径
        images: 测试图片列表
        device: GPU设备
        conf: 置信度阈值
        iou: IOU阈值
        imgsz: 图像尺寸
        warmup: 预热次数
        channels: 输入图像通道数 (RGB=3, IR=1)
        use_simotm: 预处理方法 ('RGB', 'SimOTMBBS', etc.)

    Returns:
        平均推理时间(ms)
    """
    print(f"\n{'='*60}")
    print(f"测试TensorRT Engine性能")
    print(f"{'='*60}")
    print(f"Engine文件: {engine_path}")

    # 加载TensorRT engine
    print(f"加载TensorRT engine...")
    model = YOLO(engine_path)
    print(f"✓ Engine加载成功 (将使用 channels={channels}, use_simotm={use_simotm})")

    # 预热
    print(f"预热中 ({warmup}次)...")
    for i in range(warmup):
        _ = model.predict(images[0], conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, channels=channels, use_simotm=use_simotm)

    # 正式测试
    print(f"正式测试 ({len(images)}次)...")
    times = []

    for img in images:
        torch.cuda.synchronize()  # 确保GPU操作完成
        start = time.time()
        _ = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, channels=channels, use_simotm=use_simotm)
        torch.cuda.synchronize()  # 确保GPU操作完成
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
    print(f"  吞吐量:   {1000/avg_time:.1f} fps")

    return avg_time, times

def compare_pt_vs_engine(pt_path, engine_path, images, device, conf=0.25, iou=0.7, imgsz=640, channels=3, use_simotm='RGB'):
    """
    对比PT模型和TensorRT Engine的性能

    Args:
        pt_path: .pt模型路径
        engine_path: .engine模型路径
        images: 测试图片
        device: GPU设备
        conf: 置信度阈值
        iou: IOU阈值
        imgsz: 图像尺寸
        channels: 输入图像通道数
        use_simotm: 预处理方法

    Returns:
        (pt_time, engine_time, speedup)
    """
    print(f"\n{'='*60}")
    print(f"PT vs TensorRT 性能对比")
    print(f"{'='*60}")

    # 测试PT模型
    print(f"\n[1/2] 测试PT模型...")
    pt_model = YOLO(pt_path)
    pt_times = []

    # 预热
    for i in range(10):
        _ = pt_model.predict(images[0], conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, channels=channels, use_simotm=use_simotm)

    # 测试
    for img in images:
        torch.cuda.synchronize()
        start = time.time()
        _ = pt_model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, channels=channels, use_simotm=use_simotm)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        pt_times.append(elapsed)

    pt_avg = np.mean(pt_times)
    print(f"  PT模型平均: {pt_avg:.2f}ms/张")

    # 测试Engine模型
    print(f"\n[2/2] 测试TensorRT Engine...")
    engine_model = YOLO(engine_path)
    engine_times = []

    # 预热
    for i in range(10):
        _ = engine_model.predict(images[0], conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, channels=channels, use_simotm=use_simotm)

    # 测试
    for img in images:
        torch.cuda.synchronize()
        start = time.time()
        _ = engine_model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, channels=channels, use_simotm=use_simotm)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        engine_times.append(elapsed)

    engine_avg = np.mean(engine_times)
    speedup = pt_avg / engine_avg

    print(f"  Engine平均: {engine_avg:.2f}ms/张")
    print(f"\n{'='*60}")
    print(f"性能提升:")
    print(f"{'='*60}")
    print(f"  PT模型:     {pt_avg:.2f}ms/张 ({1000/pt_avg:.1f} fps)")
    print(f"  TensorRT:   {engine_avg:.2f}ms/张 ({1000/engine_avg:.1f} fps)")
    print(f"  加速比:     {speedup:.2f}x")
    print(f"  时间节省:   {((pt_avg - engine_avg) / pt_avg * 100):.1f}%")
    print(f"{'='*60}")

    return pt_avg, engine_avg, speedup

def main():
    # ========== 配置 ==========
    MODEL_YAML = "ultralytics/cfg/models/11/yolo11m.yaml"
    MODEL_WEIGHTS = "data/Visible.pt"
    ENGINE_WEIGHTS = None  # 如果为None，会自动查找同名.engine文件
    DEVICE = "cuda:0"  # 或 "cuda:1"
    IMG_SIZE = 800
    NUM_TEST_IMAGES = 1000
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7
    BATCH_SIZES = [1, 2, 4]  # 要测试的batch sizes

    # 新增：选择测试模式
    TEST_MODE = "both"  # "pt" | "engine" | "both" | "compare"
    # "pt": 只测试PT模型
    # "engine": 只测试TensorRT engine
    # "both": 分别测试两者
    # "compare": 对比PT和Engine性能

    print("="*60)
    print("YOLO 批量推理性能测试")
    print("="*60)
    print(f"配置:")
    print(f"  模型配置: {MODEL_YAML}")
    print(f"  模型权重: {MODEL_WEIGHTS}")

    # 自动查找engine文件
    if ENGINE_WEIGHTS is None:
        auto_engine = Path(MODEL_WEIGHTS).with_suffix('.engine')
        if auto_engine.exists():
            ENGINE_WEIGHTS = str(auto_engine)
            print(f"  Engine:   {ENGINE_WEIGHTS} (自动检测)")
        else:
            print(f"  Engine:   未找到 (请先运行 tensorRT_test.py 转换)")
            if TEST_MODE in ["engine", "both", "compare"]:
                print(f"\n提示: 运行以下命令转换模型:")
                print(f"  python tensorRT_test.py {MODEL_WEIGHTS} --imgsz {IMG_SIZE}")
                if TEST_MODE != "both":
                    return
                TEST_MODE = "pt"  # 降级为只测试PT
    else:
        print(f"  Engine:   {ENGINE_WEIGHTS}")

    print(f"  设备:     {DEVICE}")
    print(f"  图片尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  测试数量: {NUM_TEST_IMAGES}张")
    print(f"  测试模式: {TEST_MODE}")
    print("="*60)

    # 检查文件是否存在
    if not Path(MODEL_WEIGHTS).exists():
        print(f"\n❌ 错误: 模型权重文件不存在: {MODEL_WEIGHTS}")
        print("请修改脚本中的 MODEL_WEIGHTS 路径")
        return

    # 生成测试图片
    test_images = create_test_images(NUM_TEST_IMAGES, size=(IMG_SIZE, IMG_SIZE))

    # 根据测试模式执行不同的测试
    if TEST_MODE == "compare":
        # 对比模式
        if ENGINE_WEIGHTS and Path(ENGINE_WEIGHTS).exists():
            compare_pt_vs_engine(
                MODEL_WEIGHTS,
                ENGINE_WEIGHTS,
                test_images,
                DEVICE,
                CONF_THRESHOLD,
                IOU_THRESHOLD,
                IMG_SIZE,
                channels=3,
                use_simotm='RGB'
            )
        else:
            print(f"\n❌ 对比模式需要Engine文件，但未找到")
        return

    elif TEST_MODE == "engine":
        # 只测试Engine
        if ENGINE_WEIGHTS and Path(ENGINE_WEIGHTS).exists():
            test_tensorrt_performance(
                ENGINE_WEIGHTS,
                test_images,
                DEVICE,
                CONF_THRESHOLD,
                IOU_THRESHOLD,
                IMG_SIZE,
                channels=3,
                use_simotm='RGB'
            )
        else:
            print(f"\n❌ Engine文件不存在")
        return

    # PT模型测试或both模式
    # 加载模型
    print(f"\n正在加载YOLO模型...")
    model = YOLO(MODEL_WEIGHTS)
    print(f"✓ 模型加载成功")

    # 检查初始显存
    print(f"\n初始GPU状态:")
    check_gpu_memory()

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
    print(f"\n实际场景分析 (27路摄像头):")
    print(f"{'-'*60}")

    for batch_size in BATCH_SIZES:
        avg_time = results[batch_size]

        # 计算24路情况下的性能
        time_per_24_frames = 27 * avg_time  # 处理24帧的总时间(ms)
        max_fps_per_camera = 1000 / time_per_24_frames  # 每路最大帧率

        # 如果使用batching，需要加上等待时间
        if batch_size > 1:
            # 假设等待窗口为batch凑齐的时间
            wait_time = (batch_size - 1) * (1000 / 27)  # 粗略估计
            total_latency = avg_time + wait_time
            print(f"Batch={batch_size}: 单张{avg_time:.1f}ms + 等待{wait_time:.0f}ms = {total_latency:.0f}ms延迟, "
                  f"27路最大{max_fps_per_camera:.2f}fps/路")
        else:
            print(f"Batch=1:  单张{avg_time:.1f}ms, 27路最大{max_fps_per_camera:.2f}fps/路")

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

    # 如果是both模式，额外测试Engine
    if TEST_MODE == "both" and ENGINE_WEIGHTS and Path(ENGINE_WEIGHTS).exists():
        print(f"\n\n{'='*60}")
        print(f"额外测试: TensorRT Engine")
        print(f"{'='*60}")

        engine_avg, _ = test_tensorrt_performance(
            ENGINE_WEIGHTS,
            test_images,
            DEVICE,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
            IMG_SIZE,
            channels=3,
            use_simotm='RGB'
        )

        # 对比PT Batch=1 vs TensorRT
        pt_batch1 = results[1]
        speedup = pt_batch1 / engine_avg

        print(f"\n{'='*60}")
        print(f"PT vs TensorRT 总结")
        print(f"{'='*60}")
        print(f"  PT:  {pt_batch1:.2f}ms/张")
        print(f"  TensorRT:      {engine_avg:.2f}ms/张")
        print(f"  加速比:        {speedup:.2f}x")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
