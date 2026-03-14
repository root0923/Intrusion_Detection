"""
TensorRT模型转换脚本

将YOLO的.pt模型转换为TensorRT .engine格式，用于加速推理

支持功能：
- 单模型转换
- 批量转换
- 多GPU支持
- FP16/FP32精度选择
- 动态batch size（可选）
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def convert_to_tensorrt(
    model_path,
    output_path=None,
    imgsz=640,
    batch=1,
    half=True,
    dynamic=False,
    workspace=4,
    device=0,
    simplify=True
):
    """
    将PT模型转换为TensorRT engine

    Args:
        model_path: .pt模型路径
        output_path: 输出.engine文件路径（可选，默认为model_path替换后缀）
        imgsz: 输入图像尺寸
        batch: batch size（如果dynamic=True，这是最大batch）
        half: 是否使用FP16精度（True更快，False精度更高）
        dynamic: 是否支持动态batch size
        workspace: TensorRT工作空间大小（GB）
        device: GPU设备ID
        simplify: 是否简化ONNX模型

    Returns:
        输出engine文件的路径
    """
    model_path = Path(model_path)

    # 检查文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 确定输出路径
    if output_path is None:
        output_path = model_path.with_suffix('.engine')
    else:
        output_path = Path(output_path)

    print("="*60)
    print("TensorRT模型转换")
    print("="*60)
    print(f"输入模型: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"配置:")
    print(f"  - 图像尺寸: {imgsz}x{imgsz}")
    print(f"  - Batch size: {batch} {'(max, dynamic)' if dynamic else '(fixed)'}")
    print(f"  - 精度: {'FP16' if half else 'FP32'}")
    print(f"  - 工作空间: {workspace}GB")
    print(f"  - GPU设备: cuda:{device}")
    print(f"  - ONNX简化: {simplify}")
    print("="*60)

    # 加载模型
    print(f"\n[1/3] 加载模型...")
    model = YOLO(model_path)
    print(f"  ✓ 模型加载成功")

    # 检测模型输入通道数
    try:
        # 获取模型第一层的输入通道数
        first_conv = None
        for module in model.model.modules():
            if hasattr(module, 'in_channels'):
                first_conv = module
                break

        if first_conv and hasattr(first_conv, 'in_channels'):
            input_channels = first_conv.in_channels
            print(f"  检测到输入通道数: {input_channels} ({'单通道/红外' if input_channels == 1 else 'RGB'})")
        else:
            input_channels = 3  # 默认RGB
            print(f"  使用默认通道数: 3 (RGB)")
    except:
        input_channels = 3
        print(f"  使用默认通道数: 3 (RGB)")

    # 导出为TensorRT engine
    print(f"\n[2/3] 转换为TensorRT engine...")
    print(f"  (这可能需要几分钟，请耐心等待...)")

    try:
        model.export(
            format='engine',
            imgsz=imgsz,
            batch=batch,
            half=half,
            dynamic=dynamic,
            workspace=workspace,
            device=device,
            simplify=simplify,
            channels=input_channels  # 关键：指定输入通道数
        )

        # ultralytics会自动生成engine文件，通常在原文件同目录
        auto_engine_path = model_path.with_suffix('.engine')

        # 如果指定了不同的输出路径，移动文件
        if auto_engine_path.exists() and auto_engine_path != output_path:
            import shutil
            shutil.move(str(auto_engine_path), str(output_path))

        print(f"  ✓ 转换成功")

    except Exception as e:
        print(f"  ✗ 转换失败: {e}")
        raise

    # 验证engine文件
    print(f"\n[3/3] 验证engine文件...")
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Engine文件已生成")
        print(f"  ✓ 文件大小: {file_size_mb:.2f} MB")
        print(f"  ✓ 文件路径: {output_path}")
    else:
        raise FileNotFoundError(f"Engine文件未生成: {output_path}")

    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)

    return output_path


def batch_convert(model_dir, pattern="*.pt", **kwargs):
    """
    批量转换目录下的所有PT模型

    Args:
        model_dir: 模型目录
        pattern: 文件匹配模式
        **kwargs: 传递给convert_to_tensorrt的其他参数
    """
    model_dir = Path(model_dir)
    model_files = sorted(model_dir.glob(pattern))

    if not model_files:
        print(f"未找到匹配的模型文件: {model_dir}/{pattern}")
        return

    print(f"找到 {len(model_files)} 个模型文件")
    print("="*60)

    success_count = 0
    failed_files = []

    for i, model_path in enumerate(model_files, 1):
        print(f"\n处理 [{i}/{len(model_files)}]: {model_path.name}")
        try:
            convert_to_tensorrt(model_path, **kwargs)
            success_count += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed_files.append(model_path.name)

    print("\n" + "="*60)
    print("批量转换汇总")
    print("="*60)
    print(f"成功: {success_count}/{len(model_files)}")
    if failed_files:
        print(f"失败文件:")
        for name in failed_files:
            print(f"  - {name}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='将YOLO PT模型转换为TensorRT Engine')
    parser.add_argument('model', type=str, help='PT模型路径或目录')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出engine文件路径')
    parser.add_argument('--imgsz', type=int, default=800, help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size (默认: 1)')
    parser.add_argument('--fp32', action='store_true', help='使用FP32精度（默认FP16）')
    parser.add_argument('--dynamic', action='store_true', help='启用动态batch size')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT工作空间大小GB (默认: 4)')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID (默认: 0)')
    parser.add_argument('--no-simplify', action='store_true', help='不简化ONNX模型')
    parser.add_argument('--batch-convert', action='store_true', help='批量转换目录下的所有PT文件')
    parser.add_argument('--pattern', type=str, default='*.pt', help='批量转换时的文件匹配模式')

    args = parser.parse_args()

    kwargs = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'half': not args.fp32,
        'dynamic': args.dynamic,
        'workspace': args.workspace,
        'device': args.device,
        'simplify': not args.no_simplify
    }

    if args.batch_convert:
        # 批量转换模式
        batch_convert(args.model, pattern=args.pattern, **kwargs)
    else:
        # 单文件转换模式
        convert_to_tensorrt(args.model, output_path=args.output, **kwargs)


if __name__ == "__main__":
    # 如果直接运行，使用默认配置
    import sys

    if len(sys.argv) == 1:
        # 示例：转换单个模型
        print("示例用法:")
        print("=" * 60)
        print("1. 转换单个模型:")
        print("   python tensorRT_test.py data/LLVIP-yolo11m-e300-16-pretrained.pt")
        print()
        print("2. 指定参数:")
        print("   python tensorRT_test.py model.pt --imgsz 800 --batch 4 --fp32")
        print()
        print("3. 批量转换:")
        print("   python tensorRT_test.py data/ --batch-convert --pattern '*.pt'")
        print()
        print("4. 多GPU:")
        print("   python tensorRT_test.py model.pt --device 1")
        print()
        print("5. 动态batch:")
        print("   python tensorRT_test.py model.pt --dynamic --batch 8")
        print("=" * 60)

        # 示例转换（注释掉，避免误操作）
        # model_path = "data/LLVIP-yolo11m-e300-16-pretrained.pt"
        # if Path(model_path).exists():
        #     convert_to_tensorrt(
        #         model_path=model_path,
        #         imgsz=800,
        #         batch=1,
        #         half=True,
        #         device=0
        #     )
    else:
        main()
