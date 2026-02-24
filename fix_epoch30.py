#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 epoch30.pt 中提取 EMA 模型，生成可用于推理的权重文件
"""

from ultralytics.utils.torch_utils import strip_optimizer
from pathlib import Path

if __name__ == "__main__":
    # 源文件
    epoch30_path = Path("runs/finetuneNegV2/lake-yolo11m-finetuneNegV22/weights/epoch60.pt")

    # 输出文件（可选，如果不指定则覆盖原文件）
    output_path = Path("runs/finetuneNegV2/lake-yolo11m-finetuneNegV22/weights/epoch60.pt")

    print(f"正在处理 {epoch30_path}...")
    print("从 EMA 中提取模型并优化...")

    # 调用 strip_optimizer 提取 EMA 模型
    result = strip_optimizer(f=epoch30_path, s=str(output_path))

    if result:
        print(f"\n✓ 成功! 模型已保存到: {output_path}")
        print("现在可以用这个文件进行推理了")
    else:
        print("\n✗ 处理失败")
