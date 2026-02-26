#!/usr/bin/env python3
"""
清理标签文件并创建缺失的空标签文件

该脚本执行以下操作：
1. 删除labels目录中多余的txt文件（没有对应图片）
2. 为extracted_frames中有图片但labels中缺少txt文件的情况创建空txt文件
"""

import os
from pathlib import Path


def clean_and_create_labels(extracted_frames_dir, labels_dir):
    """
    清理标签文件并创建缺失的空标签文件
    
    Args:
        extracted_frames_dir (str): 包含提取帧图片的目录
        labels_dir (str): 包含标签txt文件的目录
    """
    # 获取extracted_frames目录中的所有图片文件名（不含扩展名）
    image_files = set()
    for file in os.listdir(extracted_frames_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_name = Path(file).stem  # 获取不带扩展名的文件名
            image_files.add(image_name)
    
    # 获取labels目录中的所有txt文件名（不含扩展名）
    label_files = set()
    for file in os.listdir(labels_dir):
        if file.lower().endswith('.txt') and file != 'classes.txt':  # 排除classes.txt
            label_name = Path(file).stem
            label_files.add(label_name)
    
    print(f"Found {len(image_files)} images in extracted_frames")
    print(f"Found {len(label_files)} label files in labels")
    
    # 找出labels中多余的txt文件（没有对应图片）
    redundant_labels = label_files - image_files
    print(f"Found {len(redundant_labels)} redundant label files")
    
    # 删除多余的txt文件
    for label_name in redundant_labels:
        txt_file = os.path.join(labels_dir, f"{label_name}.txt")
        if os.path.exists(txt_file):
            os.remove(txt_file)
            print(f"Removed redundant label file: {txt_file}")
    
    # # 找出extracted_frames中有图片但labels中缺少txt文件的情况
    # missing_labels = image_files - label_files
    # print(f"Found {len(missing_labels)} missing label files")
    
    # # 创建缺失的空txt文件
    # for image_name in missing_labels:
    #     txt_file = os.path.join(labels_dir, f"{image_name}.txt")
    #     with open(txt_file, 'w') as f:
    #         # 创建空文件作为负样本标记
    #         pass
    #     print(f"Created empty label file: {txt_file}")
    
    print("Processing completed!")


if __name__ == "__main__":
    extracted_frames_path = "/home/ysy/object_detection/intrusion/Intrusion_Detection/data/dataset/with_neg/images/train"
    labels_path = "/home/ysy/object_detection/intrusion/Intrusion_Detection/data/dataset/with_neg/labels/train"
    
    clean_and_create_labels(extracted_frames_path, labels_path)