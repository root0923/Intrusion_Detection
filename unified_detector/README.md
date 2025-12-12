# 统一检测框架 - Unified Detection Framework

## 概述

统一检测框架是一个高效的多摄像头、多规则入侵检测系统，支持以下三种算法：
- **区域入侵检测** (Area Intrusion)
- **绊线入侵检测** (Tripwire Intrusion)
- **涉水安全检测** (Water Safety)

### 核心优势

1. **性能优化**：每个摄像头只推理一次，检测结果共享给多个规则，相比原方案性能提升约3倍
2. **热更新**：支持配置热更新，无需重启进程
3. **多进程架构**：稳定可靠，单个进程崩溃不影响其他摄像头
4. **统一管理**：一个框架管理所有检测算法，简化部署和维护

## 目录结构

```
unified_detector/
├── main.py                    # 主程序（多进程管理）
├── start_detection.bat        # 启动脚本
├── README.md                  # 使用说明
│
├── core/                      # 核心组件
│   ├── api_client.py          # API客户端
│   ├── detector.py            # 统一检测器（封装YOLO+跟踪）
│   └── processor.py           # 摄像头处理器（子进程逻辑）
│
├── rules/                     # 规则引擎
│   ├── base.py                # 抽象基类
│   ├── area_intrusion.py      # 区域入侵规则
│   ├── tripwire.py            # 绊线入侵规则
│   └── water_safety.py        # 涉水安全规则
│
├── utils/                     # 工具函数
│   ├── config_parser.py       # 配置解析器
│   └── geometry.py            # 几何工具
│
└── log/                       # 日志目录
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (如使用GPU)
- 依赖库：ultralytics, opencv-python, pycryptodome

### 2. 安装依赖

```bash
# 已在主项目环境中安装，无需额外安装
```

### 3. 配置启动脚本

编辑 `start_detection.bat`，修改以下配置：

```bat
set API_URL=http://your-server:8080
set USERNAME=your_username
set PASSWORD=your_password
set MODEL_YAML=ultralytics/cfg/models/11/yolo11x.yaml
set WEIGHTS=data/your_model_weights.pt
set DEVICE=cuda:0
```

### 4. 启动系统

```bash
# Windows
cd D:\intrusion\Intrusion_Detection
unified_detector\start_detection.bat

# Linux
python unified_detector/main.py --api-url http://localhost:8080 \
                                --username admin \
                                --password admin123 \
                                --model-yaml ultralytics/cfg/models/11/yolo11x.yaml \
                                --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
                                --device cuda:0
```

## 核心机制

### 1. 统一推理

每个摄像头只推理一次（`model.track()`），检测结果共享给所有规则：

```
摄像头1 → YOLO推理（1次）→ 检测结果 → 区域入侵规则
                                    → 绊线入侵规则
                                    → 涉水安全规则
```

### 2. 置信度过滤

- **推理阶段**：统一使用 `conf=0.25`（低阈值，保留更多候选）
- **规则阶段**：每个规则根据API配置的 `sensitivity` 过滤检测结果

### 3. 容忍时间机制

仅对区域入侵和涉水安全生效（绊线入侵不使用）：

- 检测不到目标后，等待 `tolerance_time`（默认3秒）再重置状态
- 防止短暂遮挡导致误判

### 4. 配置热更新

- 主进程每30秒（可配置）轮询API获取最新配置
- 通过Queue发送新配置到子进程，无需重启
- 支持动态启用/禁用规则

### 5. 进程生命周期

- **启动条件**：至少有一个规则启用
- **停止条件**：所有规则都撤防（`izEnable=0`）
- **更新策略**：配置变化时，通过热更新机制刷新规则

## 配置说明

### API配置格式

API接口 `/artificial/api/listDeviceAndChannel` 返回的配置格式：

```json
{
  "result": [
    {
      "deviceId": "device_001",
      "deviceName": "摄像头1",
      "deviceCode": "CAM001",
      "deviceIp": "192.168.1.100",
      "deviceChannelVos": [
        {
          "channelId": "channel_001",
          "channelName": "通道1",
          "channelCode": "CH001",
          "algorithmRules": [
            {
              "algorithmCode": "area_intrusion",
              "izEnable": "1",
              "sensitivity": 5,
              "firstAlarmTime": 1.0,
              "repeatedAlarmTime": 30.0,
              "width": 1920,
              "height": 1080,
              "algorithmRulePoints": [
                {
                  "groupType": "polygon",
                  "pointStr": "[[100,200],[400,200],[400,500],[100,500]]"
                }
              ]
            },
            {
              "algorithmCode": "tripwire_intrusion",
              "izEnable": "1",
              "sensitivity": 5,
              "repeatedAlarmTime": 30.0,
              "direction": "bidirectional",
              "width": 1920,
              "height": 1080,
              "algorithmRulePoints": [
                {
                  "groupType": "polyline",
                  "pointStr": "[[100,300],[600,300]]"
                }
              ]
            },
            {
              "algorithmCode": "water_safety",
              "izEnable": "0",
              ...
            }
          ]
        }
      ]
    }
  ]
}
```

### 报警API格式

报警上传接口 `/artificial/api/alarm`，POST请求体：

```json
{
  "deviceId": "device_001",
  "deviceName": "摄像头1",
  "deviceCode": "CAM001",
  "deviceIp": "192.168.1.100",
  "channelId": "channel_001",
  "channelName": "通道1",
  "channelCode": "CH001",
  "alarmPicCode": "base64_encoded_image",
  "nodeType": "2",
  "alarmDate": "2025-12-12 10:30:00",
  "alarmType": "area_intrusion",  // 或 tripwire_intrusion, water_safety
  "alarmTypeName": "区域入侵"     // 或 绊线入侵, 涉水检测
}
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--api-url` | str | 必填 | 后端API基础URL |
| `--username` | str | 必填 | 登录用户名 |
| `--password` | str | 必填 | 登录密码 |
| `--model-yaml` | str | yolo11x.yaml | 模型配置文件 |
| `--weights` | str | LLVIP_IF-yolo11x.pt | 模型权重文件 |
| `--device` | str | cuda:0 | 推理设备 |
| `--target-size` | int | 640 | 推理图像尺寸 |
| `--process-fps` | float | 10.0 | 处理帧率（抽帧） |
| `--tolerance-time` | float | 3.0 | 容忍时间（秒） |
| `--tracker` | str | bytetrack | 跟踪器类型 |
| `--config-update-interval` | int | 30 | 配置更新间隔（秒） |
| `--log-dir` | str | log/ | 日志目录 |

## 日志

日志文件位置：`unified_detector/log/unified_detector_YYYYMMDD.log`

日志级别：
- **INFO**：正常运行日志（启动、配置更新、报警等）
- **DEBUG**：详细调试信息
- **ERROR**：错误信息

## 故障排查

### 1. 登录失败

- 检查API URL是否正确
- 检查用户名/密码是否正确
- 检查网络连接

### 2. 模型加载失败

- 检查模型文件路径是否存在
- 检查CUDA环境是否正确配置
- 检查显存是否充足

### 3. 视频流打开失败

- 检查摄像头RTSP地址是否正确
- 检查网络连接
- 检查摄像头是否在线

### 4. 进程异常退出

- 查看日志文件中的错误信息
- 检查显存/内存是否充足
- 检查是否有异常的配置数据

## 性能优化

### 1. 调整处理帧率

```bat
set PROCESS_FPS=10.0   # 降低可减少CPU/GPU负载
```

### 2. 调整推理尺寸

```bat
set TARGET_SIZE=640    # 降低可加快推理速度，但可能影响精度
```

### 3. 使用更小的模型

```bat
set MODEL_YAML=ultralytics/cfg/models/11/yolo11n.yaml
set WEIGHTS=data/yolo11n.pt
```

## 与原代码的对比

| 特性 | 原实现 | 新框架 |
|-----|--------|--------|
| 推理次数 | 最多9次 | 最多3次 |
| 进程数 | 最多9个 | 最多3个 |
| 配置更新 | 重启进程 | 热更新 |
| 代码复用 | 三份独立代码 | 统一框架 |
| 扩展性 | 困难 | 插件式 |
| ROI处理 | 裁剪后推理 | 全图推理+后处理 |

## 开发指南

### 添加新规则

1. 在 `rules/` 目录创建新文件（如 `new_rule.py`）
2. 继承 `RuleEngine` 基类
3. 实现必要的方法：
   - `_init_rule_specific()`
   - `process()`
   - `reset()`
   - `update_config()`

4. 在 `processor.py` 中注册新规则

示例：
```python
from .base import RuleEngine

class NewRule(RuleEngine):
    def _init_rule_specific(self):
        # 初始化规则特定配置
        pass

    def process(self, frame, detections, timestamp):
        # 处理逻辑
        return alarm_info or None

    def reset(self):
        # 重置状态
        pass

    def update_config(self, new_config):
        # 更新配置
        pass
```

## 许可证

本项目与主项目使用相同的许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 查看主项目README
- 提交Issue
