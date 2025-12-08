"""
Tripwire Intrusion Detection System
基于YOLOv11-RGBT的绊线入侵检测系统
使用Ultralytics内置跟踪器（ByteTrack/BoT-SORT）
"""

from .geometry import check_line_intersection, compute_crossing_direction
from .tripwire_monitor import TripwireMonitor
from .visualizer import TripwireVisualizer

__version__ = "2.0.0"
__all__ = [
    'check_line_intersection',
    'compute_crossing_direction',
    'TripwireMonitor',
    'TripwireVisualizer'
]
