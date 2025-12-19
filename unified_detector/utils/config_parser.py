"""
配置解析器 - Configuration Parser

功能：
- 解析API配置数据
- 合并同一摄像头的三个算法规则
- 坐标转换（前端 → 实际视频流）
- 配置比较（支持热更新）
"""
import json
import logging
import numpy as np
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)


class ConfigParser:
    """配置解析器：解析设备配置、合并算法规则、坐标转换"""

    # 灵敏度映射（接口返回1-10，映射到置信度阈值）
    SENSITIVITY_MAPPING = {
        1: 0.85,   # 最低灵敏度: 非常严格
        2: 0.75,
        3: 0.65,
        4: 0.55,
        5: 0.45,   # 中等灵敏度
        6: 0.35,
        7: 0.25,
        8: 0.20,
        9: 0.15,
        10: 0.10,  # 最高灵敏度: 非常宽松
    }

    @staticmethod
    def parse_device_config(config_data: Dict) -> Dict[str, Dict]:
        """
        解析设备配置，合并同一摄像头的三个算法规则

        Args:
            config_data: 从API获取的设备配置

        Returns:
            Dict[str, Dict]: 摄像头配置字典
                key: f"{device_id}_{channel_id}"
                value: 摄像头配置（包含三个算法规则）
        """
        camera_configs = {}

        try:
            if 'result' not in config_data:
                logger.warning("配置数据中未找到 'result' 字段")
                return camera_configs

            devices = config_data['result']
            if not isinstance(devices, list):
                devices = [devices]

            for device in devices:
                device_id = device.get('deviceId', '')
                device_name = device.get('deviceName', '')
                device_code = device.get('deviceCode', '')
                device_ip = device.get('deviceIp', '')

                # 获取通道列表
                channels = device.get('deviceChannelVos', [])

                for channel in channels:
                    channel_id = channel.get('channelId', '')
                    channel_name = channel.get('channelName', '')
                    channel_code = channel.get('channelCode', '')

                    # 摄像头唯一标识
                    camera_key = f"{device_code}_{channel_code}"

                    # 初始化摄像头配置
                    if camera_key not in camera_configs:
                        camera_configs[camera_key] = {
                            'camera_key': camera_key,
                            'device_id': device_id,
                            'device_name': device_name,
                            'device_code': device_code,
                            'device_ip': device_ip,
                            'channel_id': channel_id,
                            'channel_name': channel_name,
                            'channel_code': channel_code,
                            'rules': {}
                        }

                    # 获取算法规则列表
                    algorithm_rules = channel.get('algorithmRules', [])

                    # 解析三种算法规则
                    for rule in algorithm_rules:
                        algorithm_code = rule.get('algorithmCode')

                        if algorithm_code == 'area_intrusion':
                            ConfigParser._parse_area_intrusion(rule, camera_configs[camera_key])
                        elif algorithm_code == 'tripwire_intrusion':
                            ConfigParser._parse_tripwire_intrusion(rule, camera_configs[camera_key])
                        elif algorithm_code == 'water_safety':
                            ConfigParser._parse_water_safety(rule, camera_configs[camera_key])

            # 过滤：只保留至少有一个规则启用的摄像头
            enabled_cameras = {
                key: config for key, config in camera_configs.items()
                if any(rule.get('enabled', False) for rule in config['rules'].values())
            }

            logger.info(f"✓ 解析配置成功: {len(enabled_cameras)} 个摄像头，"
                       f"共 {sum(len(c['rules']) for c in enabled_cameras.values())} 个规则")

            return enabled_cameras

        except Exception as e:
            logger.error(f"✗ 解析设备配置异常: {e}", exc_info=True)
            return {}

    @staticmethod
    def _parse_area_intrusion(rule: Dict, camera_config: Dict):
        """解析区域入侵规则"""
        is_enable = rule.get('izEnable', '0')
        if is_enable == '0':
            return

        # 提取配置
        sensitivity = int(rule.get('sensitivity', 3))
        sensitivity = ConfigParser.SENSITIVITY_MAPPING.get(sensitivity, 0.65)
        first_alarm_time = float(rule.get('firstAlarmTime', 1.0))
        repeated_alarm_time = float(rule.get('repeatedAlarmTime', 30.0))
        frontend_width = int(rule.get('width', 1920))
        frontend_height = int(rule.get('height', 1080))

        # 解析ROI点位
        roi_list = []
        algorithm_rule_points = rule.get('algorithmRulePoints', [])
        for point_item in algorithm_rule_points:
            if point_item.get('groupType') != 'polygon':
                continue
            point_str = point_item.get('pointStr', '')
            if point_str:
                try:
                    points = json.loads(point_str)
                    # 统一转换为三维列表 [region1, region2, ...]
                    if points and isinstance(points[0][0], list):
                        roi_list.extend(points)
                    else:
                        roi_list.append(points)
                except json.JSONDecodeError as e:
                    logger.error(f"解析ROI点位失败: {point_str} - {e}")

        if not roi_list:
            logger.warning(f"区域入侵规则没有有效的ROI配置: {camera_config['device_name']}/{camera_config['channel_name']}")
            return

        # 存储规则配置
        camera_config['rules']['area_intrusion'] = {
            'enabled': True,
            'sensitivity': sensitivity,
            'first_alarm_time': first_alarm_time,
            'repeated_alarm_time': repeated_alarm_time,
            'frontend_width': frontend_width,
            'frontend_height': frontend_height,
            'roi_list': roi_list,  # 前端坐标，后续需要转换
            'device_info': {
                'deviceId': camera_config['device_id'],
                'deviceName': camera_config['device_name'],
                'deviceCode': camera_config['device_code'],
                'deviceIp': camera_config['device_ip'],
                'channelId': camera_config['channel_id'],
                'channelName': camera_config['channel_name'],
                'channelCode': camera_config['channel_code']
            }
        }

        logger.debug(f"✓ 解析区域入侵规则: {camera_config['device_name']}/{camera_config['channel_name']} "
                    f"(ROI数: {len(roi_list)})")

    @staticmethod
    def _parse_tripwire_intrusion(rule: Dict, camera_config: Dict):
        """解析绊线入侵规则"""
        is_enable = rule.get('izEnable', '0')
        if is_enable == '0':
            return

        # 提取配置
        sensitivity = int(rule.get('sensitivity', 4))
        sensitivity = ConfigParser.SENSITIVITY_MAPPING.get(sensitivity, 0.55)
        repeated_alarm_time = float(rule.get('repeatedAlarmTime', 30.0))
        direction = rule.get('direction', 'bidirectional')
        frontend_width = int(rule.get('width', 1920))
        frontend_height = int(rule.get('height', 1080))

        # 解析绊线点位
        tripwire_lines = []
        algorithm_rule_points = rule.get('algorithmRulePoints', [])
        for point_item in algorithm_rule_points:
            if point_item.get('groupType') != 'polyline':
                continue
            point_str = point_item.get('pointStr', '')
            if point_str:
                try:
                    points = json.loads(point_str)
                    # 相邻点连线生成绊线：N个点 → N-1条线
                    for i in range(len(points) - 1):
                        line_points = [points[i], points[i + 1]]
                        tripwire_lines.append(line_points)
                except json.JSONDecodeError as e:
                    logger.error(f"解析绊线点位失败: {point_str} - {e}")

        if not tripwire_lines:
            logger.warning(f"绊线入侵规则没有有效的绊线配置: {camera_config['device_name']}/{camera_config['channel_name']}")
            return

        # 存储规则配置
        camera_config['rules']['tripwire_intrusion'] = {
            'enabled': True,
            'sensitivity': sensitivity,
            'repeated_alarm_time': repeated_alarm_time,
            'direction': direction,
            'frontend_width': frontend_width,
            'frontend_height': frontend_height,
            'tripwire_lines': tripwire_lines,  # 前端坐标，后续需要转换
            'device_info': {
                'deviceId': camera_config['device_id'],
                'deviceName': camera_config['device_name'],
                'deviceCode': camera_config['device_code'],
                'deviceIp': camera_config['device_ip'],
                'channelId': camera_config['channel_id'],
                'channelName': camera_config['channel_name'],
                'channelCode': camera_config['channel_code']
            }
        }

        logger.debug(f"✓ 解析绊线入侵规则: {camera_config['device_name']}/{camera_config['channel_name']} "
                    f"(绊线数: {len(tripwire_lines)})")

    @staticmethod
    def _parse_water_safety(rule: Dict, camera_config: Dict):
        """解析涉水安全规则"""
        is_enable = rule.get('izEnable', '0')
        if is_enable == '0':
            return

        # 提取配置（与区域入侵类似）
        sensitivity = int(rule.get('sensitivity', 2))
        sensitivity = ConfigParser.SENSITIVITY_MAPPING.get(sensitivity, 0.75)
        first_alarm_time = float(rule.get('firstAlarmTime', 1.0))
        repeated_alarm_time = float(rule.get('repeatedAlarmTime', 30.0))
        frontend_width = int(rule.get('width', 1920))
        frontend_height = int(rule.get('height', 1080))

        # 解析ROI点位
        roi_list = []
        algorithm_rule_points = rule.get('algorithmRulePoints', [])
        for point_item in algorithm_rule_points:
            if point_item.get('groupType') != 'polygon':
                continue
            point_str = point_item.get('pointStr', '')
            if point_str:
                try:
                    points = json.loads(point_str)
                    if points and isinstance(points[0][0], list):
                        roi_list.extend(points)
                    else:
                        roi_list.append(points)
                except json.JSONDecodeError as e:
                    logger.error(f"解析ROI点位失败: {point_str} - {e}")

        if not roi_list:
            logger.warning(f"涉水安全规则没有有效的ROI配置: {camera_config['device_name']}/{camera_config['channel_name']}")
            return

        # 存储规则配置
        camera_config['rules']['water_safety'] = {
            'enabled': True,
            'sensitivity': sensitivity,
            'first_alarm_time': first_alarm_time,
            'repeated_alarm_time': repeated_alarm_time,
            'frontend_width': frontend_width,
            'frontend_height': frontend_height,
            'roi_list': roi_list,
            'device_info': {
                'deviceId': camera_config['device_id'],
                'deviceName': camera_config['device_name'],
                'deviceCode': camera_config['device_code'],
                'deviceIp': camera_config['device_ip'],
                'channelId': camera_config['channel_id'],
                'channelName': camera_config['channel_name'],
                'channelCode': camera_config['channel_code']
            }
        }

        logger.debug(f"✓ 解析涉水安全规则: {camera_config['device_name']}/{camera_config['channel_name']} "
                    f"(ROI数: {len(roi_list)})")

    @staticmethod
    def convert_coordinates(camera_config: Dict, actual_width: int, actual_height: int):
        """
        将前端坐标转换为实际视频流坐标（就地修改）

        Args:
            camera_config: 摄像头配置
            actual_width: 实际视频流宽度
            actual_height: 实际视频流高度
        """
        for rule_type, rule_config in camera_config['rules'].items():
            if not rule_config.get('enabled', False):
                continue

            frontend_width = rule_config['frontend_width']
            frontend_height = rule_config['frontend_height']

            scale_x = actual_width / frontend_width
            scale_y = actual_height / frontend_height

            if rule_type in ['area_intrusion', 'water_safety']:
                # 转换ROI坐标
                roi_list = rule_config['roi_list']
                converted_rois = []
                for region in roi_list:
                    region_array = []
                    for point in region:
                        x, y = point
                        actual_x = int(x * scale_x)
                        actual_y = int(y * scale_y)
                        region_array.append([actual_x, actual_y])
                    converted_rois.append(np.array(region_array, dtype=np.int32))
                rule_config['roi_arrays'] = converted_rois

            elif rule_type == 'tripwire_intrusion':
                # 转换绊线坐标
                tripwire_lines = rule_config['tripwire_lines']
                converted_lines = []
                for line in tripwire_lines:
                    converted_line = []
                    for point in line:
                        x, y = point
                        actual_x = int(x * scale_x)
                        actual_y = int(y * scale_y)
                        converted_line.append([actual_x, actual_y])
                    converted_lines.append(converted_line)
                rule_config['tripwire_arrays'] = converted_lines

            logger.debug(f"坐标转换 [{rule_type}]: {frontend_width}x{frontend_height} -> "
                        f"{actual_width}x{actual_height} (scale: {scale_x:.3f}, {scale_y:.3f})")

    @staticmethod
    def compare_configs(old_configs: Dict[str, Dict], new_configs: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        比较新旧配置，找出需要添加、删除、更新的摄像头

        Args:
            old_configs: 旧配置字典
            new_configs: 新配置字典

        Returns:
            Dict: {'add': [...], 'remove': [...], 'update': [...]}
        """
        old_keys = set(old_configs.keys())
        new_keys = set(new_configs.keys())

        # 新增的摄像头
        added = list(new_keys - old_keys)

        # 删除的摄像头
        removed = list(old_keys - new_keys)

        # 更新的摄像头（配置发生变化）
        common_keys = old_keys & new_keys
        updated = []
        for key in common_keys:
            if not ConfigParser._configs_equal(old_configs[key], new_configs[key]):
                updated.append(key)

        return {
            'add': added,
            'remove': removed,
            'update': updated
        }

    @staticmethod
    def _configs_equal(config1: Dict, config2: Dict) -> bool:
        """比较两个配置是否相等（只比较关键字段）"""
        # 比较规则数量
        if set(config1['rules'].keys()) != set(config2['rules'].keys()):
            return False

        # 比较每个规则的关键配置
        for rule_type in config1['rules']:
            rule1 = config1['rules'][rule_type]
            rule2 = config2['rules'][rule_type]

            # 比较关键字段
            compare_fields = ['enabled', 'sensitivity', 'first_alarm_time', 'repeated_alarm_time',
                            'roi_list', 'tripwire_lines', 'direction']

            for field in compare_fields:
                if rule1.get(field) != rule2.get(field):
                    return False

        return True
