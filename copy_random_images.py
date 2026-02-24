#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从源目录随机选择2000张图片及其对应的标签文件，
并将其复制到指定的目标目录。
"""

import os
import random
import shutil
from pathlib import Path


def copy_random_images_and_labels(src_dir, dest_img_dir, dest_label_dir, num_images=2000):
    """
    从源目录随机选择指定数量的图片及其对应的标签文件，
    并将其复制到目标目录。
    
    Args:
        src_dir (str): 源目录路径
        dest_img_dir (str): 目标图片目录路径
        dest_label_dir (str): 目标标签目录路径
        num_images (int): 要复制的图片数量，默认为2000
    """
    # 确保目标目录存在
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    # 获取所有图片文件（支持常见图片格式）
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
    all_files = os.listdir(src_dir)
    
    # 找出所有图片文件
    img_files = []
    for file in all_files:
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in img_extensions:
            img_files.append(file)
    
    print(f"在源目录找到 {len(img_files)} 张图片")
    
    if len(img_files) < num_images:
        print(f"警告: 源目录只有 {len(img_files)} 张图片，少于请求的 {num_images} 张")
        num_images = len(img_files)
    
    # 随机选择指定数量的图片
    selected_imgs = random.sample(img_files, num_images)
    
    copied_count = 0
    
    for img_file in selected_imgs:
        # 获取图片的基本名称（不含扩展名）
        base_name = os.path.splitext(img_file)[0]
        
        # 查找对应的标签文件
        label_file = base_name + '.txt'
        src_img_path = os.path.join(src_dir, img_file)
        src_label_path = os.path.join(src_dir, label_file)
        
        # 检查标签文件是否存在
        if os.path.exists(src_label_path):
            # 复制图片文件
            dest_img_path = os.path.join(dest_img_dir, img_file)
            shutil.copy2(src_img_path, dest_img_path)
            
            # 复制标签文件
            dest_label_path = os.path.join(dest_label_dir, label_file)
            shutil.copy2(src_label_path, dest_label_path)
            
            copied_count += 1
            
            if copied_count % 100 == 0:
                print(f"已复制 {copied_count}/{num_images} 对文件")
        else:
            print(f"警告: 找不到 {img_file} 对应的标签文件 {label_file}")
    
    print(f"完成! 成功复制了 {copied_count} 对图片和标签文件")


def main():
    # 定义路径
    src_dir = "/home/ysy/object_detection/train"
    dest_img_dir = "/home/ysy/object_detection/intrusion/Intrusion_Detection/data/dataset/IR/images/train"
    dest_label_dir = "/home/ysy/object_detection/intrusion/Intrusion_Detection/data/dataset/IR/labels/train"
    
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"错误: 源目录 {src_dir} 不存在")
        return
    
    # 检查目标目录是否存在
    if not os.path.exists(dest_img_dir):
        print(f"错误: 目标图片目录 {dest_img_dir} 不存在")
        return
    
    if not os.path.exists(dest_label_dir):
        print(f"错误: 目标标签目录 {dest_label_dir} 不存在")
        return
    
    # 开始复制
    print("开始复制随机图片和标签文件...")
    copy_random_images_and_labels(src_dir, dest_img_dir, dest_label_dir, 2000)


if __name__ == "__main__":
    main()