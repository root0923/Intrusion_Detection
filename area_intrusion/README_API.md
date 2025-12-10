# 区域入侵检测系统 - 后端API对接版本使用说明

## 概述

`intrusion_detector_api.py` 是一个与后端API深度集成的多流并行检测系统，支持：

- ✅ 自动登录与保活
- ✅ 动态获取设备配置和视频流地址
- ✅ 多视频流并行检测
- ✅ 前端坐标到实际流坐标的自动转换
- ✅ 报警信息自动上传
- ✅ 配置热更新（无需重启）
- ✅ 视频流自动重连

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        主程序                                │
│  - 登录认证                                                  │
│  - 保活线程                                                  │
│  - 配置更新循环                                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   DetectionManager                           │
│  - 管理多个检测进程                                          │
│  - 启动/停止/重载检测器                                      │
└────────┬────────┬────────┬────────┬────────────────────────┘
         │        │        │        │
         ▼        ▼        ▼        ▼
    ┌────────┐┌────────┐┌────────┐┌────────┐
    │通道1   ││通道2   ││通道3   ││通道N   │
    │检测进程││检测进程││检测进程││检测进程│
    └────────┘└────────┘└────────┘└────────┘
         │        │        │        │
         └────────┴────────┴────────┘
                 │
                 ▼
         ┌──────────────┐
         │ 后端API      │
         │ - 视频流     │
         │ - 报警上传   │
         └──────────────┘
```

## 核心模块

### 1. APIClient - API客户端
- `login()` - 登录获取token
- `keep_alive()` - 保持登录状态
- `get_device_config()` - 获取设备配置
- `get_stream_url()` - 获取视频流地址
- `upload_alarm()` - 上传报警信息

### 2. ConfigManager - 配置管理器
- `parse_device_config()` - 解析设备配置，筛选区域入侵规则
- `convert_points()` - 坐标转换（前端 → 实际视频流）
- `compare_configs()` - 配置差异检测

### 3. DetectionManager - 多流管理器
- `start_detector()` - 启动单个检测进程
- `stop_detector()` - 停止检测进程
- `reload_detector()` - 重启检测进程（配置变更时）
- `stop_all()` - 停止所有检测进程

### 4. stream_detector_worker - 检测进程
- 独立进程运行，互不干扰
- 自动重连机制
- ROI区域检测
- 报警触发与上传

## 使用方法

### 基本用法

```bash
python area_intrusion/intrusion_detector_api.py \
    --api-url http://localhost:8080 \
    --username admin \
    --password admin123 \
    --model-yaml ultralytics/cfg/models/11/yolo11x.yaml \
    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
    --device cuda:0
```

### 参数说明

#### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--api-url` | 后端API基础URL | `http://192.168.1.100:8080` |
| `--username` | 登录用户名 | `admin` |
| `--password` | 登录密码 | `admin123` |

#### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-yaml` | `ultralytics/cfg/models/11/yolo11x.yaml` | 模型配置文件 |
| `--weights` | `data/LLVIP_IF-yolo11x-e300-16-pretrained.pt` | 模型权重文件 |
| `--device` | `cuda:0` | 计算设备（`cuda:0` 或 `cpu`） |
| `--target-size` | `640` | YOLO检测输入尺寸 |
| `--process-fps` | `10.0` | 每秒处理帧数（抽帧） |
| `--config-update-interval` | `30` | 配置更新间隔（秒） |

### 完整示例

```bash
# 开发/调试模式（30秒更新一次配置）
python area_intrusion/intrusion_detector_api.py \
    --api-url http://192.168.1.100:8080 \
    --username admin \
    --password admin123 \
    --model-yaml ultralytics/cfg/models/11/yolo11x.yaml \
    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
    --device cuda:0 \
    --target-size 640 \
    --process-fps 10 \
    --config-update-interval 30

# 生产模式（5分钟更新一次配置）
python area_intrusion/intrusion_detector_api.py \
    --api-url http://192.168.1.100:8080 \
    --username admin \
    --password admin123 \
    --config-update-interval 300
```

## 工作流程

### 1. 启动阶段

```
[1/5] 登录后端系统
  ↓
[2/5] 启动保活线程（每5分钟调用一次）
  ↓
[3/5] 获取初始配置
  ↓
[4/5] 启动检测管理器
  ├─ 筛选启用且布防的通道
  ├─ 为每个通道启动独立检测进程
  └─ 打印检测器状态
  ↓
[5/5] 配置更新循环
```

### 2. 单通道检测流程

```
获取视频流地址
  ↓
打开RTSP流
  ↓
获取实际视频流尺寸
  ↓
坐标转换（前端 → 实际）
  ↓
初始化ROI管理器
  ↓
初始化报警管理器
  ↓
┌─────────────────┐
│  帧处理循环      │
│  1. 读取帧      │
│  2. 抽帧检测    │
│  3. ROI遮罩     │
│  4. YOLO检测    │
│  5. 入侵判断    │
│  6. 报警上传    │
└─────────────────┘
  ↓（如果流断开）
自动重连（最多5次）
```

### 3. 配置更新流程

```
定时触发（默认30秒）
  ↓
获取最新配置
  ↓
解析配置
  ↓
比对新旧配置
  ↓
┌──────────────────┐
│ 处理变更         │
│ - 新增通道 → 启动│
│ - 删除通道 → 停止│
│ - 更新通道 → 重启│
└──────────────────┘
  ↓
更新当前配置
```

## 坐标转换机制

### 问题背景

前端配置的区域点位是基于**前端显示分辨率**（如 1920x1080），但实际视频流可能是其他分辨率（如 1280x720）。直接使用前端坐标会导致检测区域错位。

### 解决方案

自动获取实际视频流分辨率，并按比例转换点位坐标：

```python
# 前端配置
frontend_width = 1920
frontend_height = 1080
frontend_points = [[100, 200], [300, 200], [300, 400], [100, 400]]

# 实际视频流
actual_width = 1280
actual_height = 720

# 计算缩放比例
scale_x = actual_width / frontend_width    # 1280 / 1920 = 0.667
scale_y = actual_height / frontend_height  # 720 / 1080 = 0.667

# 转换坐标
actual_points = []
for x, y in frontend_points:
    actual_x = int(x * scale_x)  # 100 * 0.667 = 67
    actual_y = int(y * scale_y)  # 200 * 0.667 = 133
    actual_points.append([actual_x, actual_y])
```

## 报警机制

### 报警条件

1. **置信度过滤**：检测置信度 ≥ `sensitivity`（从后端配置获取）
2. **持续时间消抖**：目标持续出现时间 ≥ `firstAlarmTime`
3. **重复报警间隔**：距离上次报警 ≥ `repeatedAlarmTime`

### 报警流程

```
检测到人员（置信度 ≥ 阈值）
  ↓
记录首次入侵时间
  ↓
持续检测...
  ↓
持续时间 ≥ firstAlarmTime？
  ↓ 是
距离上次报警 ≥ repeatedAlarmTime？
  ↓ 是
触发报警
  ├─ 可视化检测结果
  ├─ 缩放到前端显示尺寸
  ├─ Base64编码
  └─ 上传到后端API
```

### 报警数据格式

```json
{
  "deviceId": "1996128063770456065",
  "deviceName": "34020000001110000001",
  "deviceCode": "走廊过道3",
  "deviceIp": "192.128.22.15",
  "channelId": "1996128137292410881",
  "channelName": "走廊过道3",
  "channelCode": "1996128063770456065",
  "alarmPicCode": "base64_encoded_image...",
  "nodeType": "2",
  "alarmDate": "2025-12-09 15:30:00",
  "alarmType": "area_intrusion",
  "alarmTypeName": "区域入侵"
}
```

## 配置热更新

系统会定期检查后端配置变化，并自动应对：

| 变化类型 | 处理方式 | 说明 |
|---------|---------|------|
| 新增通道 | 启动新的检测进程 | 自动获取流地址并开始检测 |
| 删除通道 | 停止对应检测进程 | 优雅关闭，释放资源 |
| 配置更新 | 重启检测进程 | 停止旧进程，启动新进程应用新配置 |

**比较的配置字段**：
- `sensitivity` - 置信度阈值
- `first_alarm_time` - 首次报警时间
- `repeated_alarm_time` - 重复报警间隔
- `frontend_width` - 前端宽度
- `frontend_height` - 前端高度
- `point_list` - 区域点位
- `is_enable` - 是否启用

## 自动重连机制

当视频流断开时，检测进程会自动尝试重连：

- **最大重试次数**：5次
- **重连间隔**：5秒
- **重试策略**：指数退避（可选）

```python
retry_count = 0
max_retries = 5
retry_delay = 5

while not stop_event.is_set() and retry_count < max_retries:
    try:
        cap = cv2.VideoCapture(stream_url)
        # ... 检测流程 ...
        retry_count = 0  # 成功后重置计数
    except Exception:
        retry_count += 1
        time.sleep(retry_delay)
```

## 日志系统

使用 Python `logging` 模块，日志格式：

```
2025-12-09 15:30:00 [INFO] ✓ 登录成功
2025-12-09 15:30:01 [INFO] 保活线程启动（间隔: 300s）
2025-12-09 15:30:02 [INFO] ✓ 解析通道配置: 设备A/通道1 (区域数: 2)
2025-12-09 15:30:03 [INFO] ✓ 启动检测器: 设备A/通道1
2025-12-09 15:30:10 [INFO] 🚨 报警触发! (持续 2.5s, 检测数: 1)
```

### 日志级别

- `INFO` - 重要操作和状态变化
- `DEBUG` - 详细调试信息（需修改代码启用）
- `WARNING` - 警告信息（如保活失败）
- `ERROR` - 错误信息（如登录失败、配置解析错误）

## 性能优化

### 1. 抽帧处理

通过 `--process-fps` 参数控制检测频率，降低GPU负载：

```bash
# 每秒处理2帧（适合高分辨率流）
--process-fps 2

# 每秒处理10帧（默认，平衡性能与实时性）
--process-fps 10

# 每秒处理30帧（高实时性要求）
--process-fps 30
```

### 2. 多GPU支持

使用不同的GPU设备运行不同的检测进程：

```bash
# 通道1-5使用GPU 0
--device cuda:0

# 通道6-10使用GPU 1（需手动修改代码分配）
```

### 3. 模型选择

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|---------|
| yolo11n | 最快 | 低 | 多流并发（>10路） |
| yolo11s | 快 | 中等 | 平衡模式（5-10路） |
| yolo11m | 中等 | 高 | 精度优先（3-5路） |
| yolo11x | 慢 | 最高 | 单流/双流 |

## 常见问题

### Q1: 如何查看检测效果？

当前版本不包含可视化窗口（用于服务器部署）。可以通过日志查看检测状态，或修改代码添加 `cv2.imshow()`。

### Q2: 如何调整报警灵敏度？

在后端管理界面调整以下参数：
- `sensitivity` - 置信度阈值（越低越灵敏）
- `firstAlarmTime` - 首次报警时间（越短越灵敏）

### Q3: 如何处理误报？

1. 提高 `sensitivity`（如 0.3 → 0.5）
2. 增加 `firstAlarmTime`（如 1s → 3s）
3. 优化ROI区域，排除干扰区域

### Q4: 视频流一直重连失败怎么办？

1. 检查网络连接
2. 验证RTSP地址是否正确
3. 检查摄像头是否在线
4. 查看日志中的错误信息

### Q5: 配置更新后没有生效？

1. 检查 `isEnable` 是否为 `true`
2. 查看日志，确认配置变化被检测到
3. 检查配置更新间隔是否过长

### Q6: CPU/GPU占用过高？

1. 降低 `--process-fps`（如 10 → 5）
2. 使用更轻量的模型（如 yolo11n）
3. 减少并发检测的通道数
4. 增加抽帧间隔

## 开发建议

### 添加新功能

1. **添加新的API接口**：在 `APIClient` 类中添加方法
2. **修改检测逻辑**：在 `stream_detector_worker` 函数中修改
3. **自定义报警条件**：在 `AlarmManager.update_intrusion` 中修改
4. **调整日志级别**：修改 `logging.basicConfig(level=...)`

### 调试技巧

```python
# 1. 启用DEBUG日志
logging.basicConfig(level=logging.DEBUG, ...)

# 2. 单通道测试（修改配置过滤条件）
if channel_id != '指定通道ID':
    continue

# 3. 保存检测帧到文件
cv2.imwrite(f'debug_frame_{frame_count}.jpg', vis_frame)
```

## 许可与支持

基于 YOLOv11-RGBT 项目开发，遵循相同的开源许可。

如有问题，请参考：
- 主项目文档：`CLAUDE.md`
- 原始检测脚本：`area_intrusion/intrusion_detector.py`
- YOLOv11-RGBT论文：https://arxiv.org/abs/2506.14696
