"""
几何工具 - Geometry Utilities

功能：
- 点在多边形判断
- 框与ROI相交判断
- 图像处理工具
"""
import cv2
import numpy as np
import base64
from typing import List, Tuple, Dict


def point_in_polygon(point: Tuple[float, float], polygon) -> bool:
    """
    判断点是否在多边形内

    Args:
        point: (x, y) 点坐标
        polygon: 多边形顶点 [[x1,y1], [x2,y2], ...] (可以是list或np.ndarray)

    Returns:
        bool: 点是否在多边形内
    """
    # 确保polygon是numpy数组
    if not isinstance(polygon, np.ndarray):
        polygon = np.array(polygon, dtype=np.float32)

    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0


def bbox_center_in_rois(bbox: List[float], rois: List[np.ndarray]) -> bool:
    """
    判断检测框中心点是否在任一ROI区域内

    Args:
        bbox: [x1, y1, x2, y2] 检测框
        rois: List[np.ndarray] ROI区域列表

    Returns:
        bool: 中心点是否在任一ROI内
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    for roi in rois:
        if point_in_polygon((cx, cy), roi):
            return True

    return False


def resize_and_encode_image(image: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, str]:
    """
    缩放并Base64编码图片

    Args:
        image: numpy array 图像
        width: 目标宽度
        height: 目标高度

    Returns:
        Tuple[np.ndarray, str]: (缩放后图像, base64编码)
    """
    if width and height:
        try:
            resized = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = image
    else:
        resized = image

    success, buffer = cv2.imencode('.jpg', resized)
    if not success:
        success, buffer = cv2.imencode('.jpg', image)

    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return resized, img_b64


def draw_rois(image: np.ndarray, rois: List[np.ndarray], color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    在图像上绘制ROI区域

    Args:
        image: 原始图像
        rois: ROI区域列表
        color: 绘制颜色
        thickness: 线条粗细

    Returns:
        np.ndarray: 绘制后的图像
    """
    img_draw = image.copy()

    for roi_id, roi in enumerate(rois):
        # 确保ROI是numpy数组
        if not isinstance(roi, np.ndarray):
            roi = np.array(roi, dtype=np.int32)

        cv2.polylines(img_draw, [roi], isClosed=True, color=color, thickness=thickness)

        # 绘制半透明填充
        overlay = img_draw.copy()
        cv2.fillPoly(overlay, [roi], color)
        cv2.addWeighted(overlay, 0.2, img_draw, 0.8, 0, img_draw)

        # 添加ROI标签
        centroid = roi.mean(axis=0).astype(int)
        cv2.putText(img_draw, f'ROI-{roi_id}', tuple(centroid),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img_draw


def draw_tripwires(image: np.ndarray, lines: List[List[List[int]]], color=(0, 255, 255), thickness=3) -> np.ndarray:
    """
    在图像上绘制绊线

    Args:
        image: 原始图像
        lines: 绊线列表 [[[x1,y1],[x2,y2]], ...]
        color: 绘制颜色
        thickness: 线条粗细

    Returns:
        np.ndarray: 绘制后的图像
    """
    img_draw = image.copy()

    for line_id, line in enumerate(lines):
        pt1 = tuple(map(int, line[0]))
        pt2 = tuple(map(int, line[1]))
        cv2.line(img_draw, pt1, pt2, color, thickness)

        # 绘制端点
        cv2.circle(img_draw, pt1, 5, (0, 0, 255), -1)
        cv2.circle(img_draw, pt2, 5, (0, 0, 255), -1)

        # 添加标签
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(img_draw, f'Line-{line_id}', (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_draw


def draw_detections(image: np.ndarray, detections: List[Dict],
                    conf_threshold: float = 0.0, class_names: Dict[int, str] = None) -> np.ndarray:
    """
    在图像上绘制检测框

    Args:
        image: 原始图像
        detections: 检测结果列表
        conf_threshold: 置信度阈值（用于区分颜色）
        class_names: 类别名称字典

    Returns:
        np.ndarray: 绘制后的图像
    """
    vis_image = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['conf']
        cls = det['cls']

        # 根据置信度选择颜色
        is_high_conf = conf >= conf_threshold
        color = (0, 0, 255) if is_high_conf else (255, 144, 30)

        # 绘制框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 标签
        if class_names:
            cls_name = class_names.get(cls, str(cls))
        else:
            cls_name = str(cls)
        label = f'{cls_name} {conf:.2f}'

        # 标签背景
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # 标签文字
        cv2.putText(vis_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_image


def draw_alarm_text(image: np.ndarray, text: str, position=(10, 40)) -> np.ndarray:
    """
    在图像上绘制报警文字

    Args:
        image: 原始图像
        text: 报警文字
        position: 文字位置

    Returns:
        np.ndarray: 绘制后的图像
    """
    img_draw = image.copy()
    cv2.putText(img_draw, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return img_draw
