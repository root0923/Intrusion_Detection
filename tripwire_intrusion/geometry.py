"""
几何工具函数
提供线段相交判定、方向判断等功能
"""

import numpy as np
from typing import Tuple, Optional


def check_line_intersection(line_p1: Tuple[float, float],
                            line_p2: Tuple[float, float],
                            track_p1: Tuple[float, float],
                            track_p2: Tuple[float, float]) -> bool:
    """
    检查两条线段是否相交

    使用向量叉积方法判断线段相交

    Args:
        line_p1: 绊线起点 (x, y)
        line_p2: 绊线终点 (x, y)
        track_p1: 轨迹线段起点 (x, y)
        track_p2: 轨迹线段终点 (x, y)

    Returns:
        bool: 是否相交
    """
    x1, y1 = line_p1
    x2, y2 = line_p2
    x3, y3 = track_p1
    x4, y4 = track_p2

    # 计算行列式
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 平行或共线
    if abs(denom) < 1e-10:
        return False

    # 计算参数t和u
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # 两个参数都在[0,1]区间内，表示线段相交
    return 0 <= t <= 1 and 0 <= u <= 1


def compute_crossing_direction(line_p1: Tuple[float, float],
                               line_p2: Tuple[float, float],
                               track_prev: Tuple[float, float],
                               track_curr: Tuple[float, float],
                               image_height: int = None) -> Optional[str]:
    """
    计算目标穿越绊线的方向

    使用叉积判断目标相对于绊线的位置变化
    为了符合传统数学直觉，将图像坐标系转换为左下角为原点（Y轴向上）再计算

    Args:
        line_p1: 绊线起点 (x, y) - 图像坐标系
        line_p2: 绊线终点 (x, y) - 图像坐标系
        track_prev: 目标前一位置 (x, y) - 图像坐标系
        track_curr: 目标当前位置 (x, y) - 图像坐标系
        image_height: 图像高度（用于坐标转换），如果为None则使用图像坐标系直接计算

    Returns:
        str or None: 穿越方向
            - 'left-to-right': 从左侧到右侧
            - 'right-to-left': 从右侧到左侧
            - None: 未穿越
    """
    # 如果提供了图像高度，转换到传统坐标系（左下角为原点，Y轴向上）
    if image_height is not None:
        # 转换坐标：y_new = image_height - y_old
        p1 = (line_p1[0], image_height - line_p1[1])
        p2 = (line_p2[0], image_height - line_p2[1])
        prev = (track_prev[0], image_height - track_prev[1])
        curr = (track_curr[0], image_height - track_curr[1])
    else:
        # 直接使用图像坐标系
        p1 = line_p1
        p2 = line_p2
        prev = track_prev
        curr = track_curr

    # 绊线向量
    line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    # 计算前一位置相对于线段的叉积
    vec_to_prev = np.array([prev[0] - p1[0], prev[1] - p1[1]])
    cross_prev = np.cross(line_vec, vec_to_prev)

    # 计算当前位置相对于线段的叉积
    vec_to_curr = np.array([curr[0] - p1[0], curr[1] - p1[1]])
    cross_curr = np.cross(line_vec, vec_to_curr)

    # 符号相反表示穿越了绊线
    if cross_prev * cross_curr < 0:
        # cross_prev > 0: 前一位置在左侧
        # cross_curr < 0: 当前位置在右侧
        # 则是从左到右穿越
        if cross_prev > 0:
            return 'left-to-right'
        else:
            return 'right-to-left'

    return None


def point_to_line_distance(point: Tuple[float, float],
                           line_p1: Tuple[float, float],
                           line_p2: Tuple[float, float]) -> float:
    """
    计算点到线段的距离

    Args:
        point: 点坐标 (x, y)
        line_p1: 线段起点 (x, y)
        line_p2: 线段终点 (x, y)

    Returns:
        float: 距离
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    # 线段长度
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if line_len_sq < 1e-10:
        # 线段退化为点
        return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    # 计算投影参数t
    t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_len_sq))

    # 投影点
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    # 距离
    return np.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)


def get_bbox_bottom_center(bbox: list) -> Tuple[float, float]:
    """
    获取检测框底部中心点（更接近人的地面位置）

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (x_center, y_bottom)
    """
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0
    y_bottom = y2
    return (x_center, y_bottom)


def get_bbox_center(bbox: list) -> Tuple[float, float]:
    """
    获取检测框中心点

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (x_center, y_center)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
