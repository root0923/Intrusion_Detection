"""
统一检测框架 - Unified Detection Framework

支持多摄像头、多规则的入侵检测系统
- 区域入侵检测 (Area Intrusion)
- 绊线入侵检测 (Tripwire Intrusion)
- 涉水安全检测 (Water Safety)

核心优势：
1. 每个摄像头只推理一次，结果共享给多个规则
2. 支持配置热更新，无需重启进程
3. 多进程架构，稳定可靠
"""

__version__ = '1.0.0'
__author__ = 'Intrusion Detection Team'
