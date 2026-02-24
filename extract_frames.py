#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频帧提取脚本
此脚本用于从MP4视频中每秒抽取一帧并保存为图片，以便后续用于YOLO模型的训练
"""

import os
import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, fps_rate=1):
    """
    从视频中按指定帧率提取帧
    
    Args:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出图片保存目录
        fps_rate (int): 每秒提取的帧数，默认为1（即每秒1帧）
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps
    
    print(f"视频总帧数: {total_frames}")
    print(f"视频FPS: {video_fps}")
    print(f"视频时长: {duration:.2f} 秒")
    print(f"将按每秒 {fps_rate} 帧的频率提取图片")
    
    # 计算每隔多少帧提取一次
    frame_interval = int(video_fps / fps_rate)
    
    count = 0  # 总提取帧数计数器
    frame_number = 0  # 当前帧号
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 检查是否应该保存当前帧
        if frame_number % frame_interval == 0:
            # 生成输出文件名，格式为: video_name_frame_00001.jpg
            filename = f"{Path(video_path).stem}_frame_{count+1:05d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # 保存图片
            success = cv2.imwrite(output_path, frame)
            if success:
                print(f"已保存: {output_path}")
                count += 1
            else:
                print(f"保存失败: {output_path}")
        
        frame_number += 1
    
    cap.release()
    print(f"\n完成！总共提取了 {count} 帧图片到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="从MP4视频中每秒抽取一帧并保存为图片")
    parser.add_argument("--video", "-v", type=str, required=True, 
                        help="输入的MP4视频文件路径")
    parser.add_argument("--output", "-o", type=str, default="./extracted_frames", 
                        help="输出图片保存目录 (默认: ./extracted_frames)")
    parser.add_argument("--fps", "-f", type=int, default=3, 
                        help="每秒提取的帧数 (默认: 1，即每秒1帧)")
    
    args = parser.parse_args()
    
    # 检查输入视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误：视频文件 {args.video} 不存在")
        return
    
    # 检查视频文件扩展名
    if not args.video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
        print(f"警告：文件 {args.video} 可能不是视频文件")
    
    extract_frames(args.video, args.output, args.fps)


if __name__ == "__main__":
    main()