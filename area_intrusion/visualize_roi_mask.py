import cv2
import numpy as np
import json
import os

def load_rois_config(filepath="rois_config.json"):
    """从文件加载ROI配置"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"配置文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 将列表转换回元组
    rois = [[tuple(point) for point in roi] for roi in config['rois']]
    return rois, (config['image_height'], config['image_width'])

def create_roi_mask(image_shape, rois_list):
    """创建ROI蒙版 (ROI区域为255，其他为0)"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for roi_points in rois_list:
        if len(roi_points) >= 3:
            points_array = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)
    return mask

def apply_roi_mask(image, mask, mode='mask_outside'):
    """
    应用ROI蒙版到图像

    参数:
        image: 输入图像
        mask: ROI蒙版 (ROI区域为255)
        mode: 'mask_inside' - 将ROI区域内变黑
              'mask_outside' - 将ROI区域外变黑 (保留ROI区域)

    返回:
        masked_image: 处理后的图像
    """
    masked_image = image.copy()

    if mode == 'mask_inside':
        # 将ROI区域内变黑
        masked_image[mask == 255] = 0
    elif mode == 'mask_outside':
        # 将ROI区域外变黑 (只保留ROI区域)
        masked_image[mask == 0] = 0
    else:
        raise ValueError(f"未知的模式: {mode}")

    return masked_image

def visualize_roi_mask(image_path, config_path="rois_config.json",
                       mode='mask_inside', save_path=None):
    """
    可视化ROI蒙版效果

    参数:
        image_path: 输入图像路径
        config_path: ROI配置文件路径
        mode: 'mask_inside' - 将ROI区域内变黑
              'mask_outside' - 将ROI区域外变黑
        save_path: 保存路径，如果为None则自动生成
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 2. 加载ROI配置
    rois, (config_h, config_w) = load_rois_config(config_path)
    print(f"加载了 {len(rois)} 个ROI区域")
    print(f"配置图像尺寸: {config_w}x{config_h}")
    print(f"当前图像尺寸: {image.shape[1]}x{image.shape[0]}")

    # 3. 如果图像尺寸不匹配，进行缩放
    if image.shape[:2] != (config_h, config_w):
        print(f"警告: 图像尺寸不匹配，将调整到配置尺寸")
        image = cv2.resize(image, (config_w, config_h))

    # 4. 创建ROI蒙版
    mask = create_roi_mask(image.shape, rois)

    # 5. 应用蒙版
    masked_image = apply_roi_mask(image, mask, mode=mode)

    # 6. 创建对比图
    # 在原图上绘制ROI边界
    original_with_roi = image.copy()
    for i, roi_points in enumerate(rois):
        points_array = np.array(roi_points, dtype=np.int32)
        cv2.polylines(original_with_roi, [points_array], True, (0, 255, 0), 2)

        # 添加ROI编号
        center_x = sum(p[0] for p in roi_points) // len(roi_points)
        center_y = sum(p[1] for p in roi_points) // len(roi_points)
        cv2.putText(original_with_roi, f"ROI-{i+1}", (center_x-30, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 并排显示
    comparison = np.hstack([original_with_roi, masked_image])

    # 7. 保存结果
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"roi_masked_{mode}_{base_name}.jpg"

    cv2.imwrite(save_path, masked_image)
    print(f"✓ 蒙版图像已保存: {save_path}")

    # 保存对比图
    comparison_path = save_path.replace('.jpg', '_comparison.jpg')
    cv2.imwrite(comparison_path, comparison)
    print(f"✓ 对比图像已保存: {comparison_path}")

    # 8. 显示结果
    cv2.namedWindow("ROI Mask Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI Mask Comparison", comparison)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return masked_image, mask

def main():
    import argparse

    parser = argparse.ArgumentParser(description='根据ROI配置可视化蒙版效果')
    parser.add_argument('--image', type=str, default='data/dataset/infrared/n286.jpg',
                       help='输入图像路径')
    parser.add_argument('--config', type=str, default='rois_config.json',
                       help='ROI配置文件路径')
    parser.add_argument('--mode', type=str, default='mask_inside',
                       choices=['mask_inside', 'mask_outside'],
                       help='蒙版模式: mask_inside=ROI内变黑, mask_outside=ROI外变黑')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径')
    parser.add_argument('--batch', type=str, default=None,
                       help='批量处理模式：输入图像目录')
    parser.add_argument('--batch_output', type=str, default=None,
                       help='批量处理输出目录')

    args = parser.parse_args()

    print("=" * 60)
    print("ROI 蒙版可视化工具")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"  - mask_inside: 将ROI区域内变黑（ROI外保留）")
    print(f"  - mask_outside: 将ROI区域外变黑（ROI内保留）")
    print("=" * 60)


    visualize_roi_mask(args.image, args.config, args.mode, args.output)

if __name__ == "__main__":
    main()
