# 绊线入侵检测系统 (Tripwire Intrusion Detection)

基于YOLOv11-RGBT的绊线入侵检测系统，支持多通道图像输入（RGB、红外、RGBT融合），提供高精度的目标跟踪和方向判定。

## 功能特性

- ✅ **多模态检测**: 支持RGB、红外、RGBT融合图像输入
- ✅ **SOTA跟踪器**: 使用Ultralytics内置跟踪器（ByteTrack/BoT-SORT），准确度高
- ✅ **绊线监控**: 支持多条绊线，每条线可独立配置方向规则
- ✅ **方向判断**: 基于向量叉积的精确方向判定（左到右、右到左、双向）
- ✅ **事件记录**: 实时记录穿越事件，支持JSON导出
- ✅ **可视化**: 实时绘制绊线、轨迹、检测框、事件提示
- ✅ **多输入支持**: 视频文件、RTSP流

## 系统架构

```
tripwire_intrusion/
├── __init__.py              # 模块初始化
├── geometry.py              # 几何工具（线段相交、方向判断）
├── tripwire_monitor.py      # 绊线监控核心逻辑
├── visualizer.py            # 可视化工具
├── tripwire_detector.py     # 主检测脚本（使用Ultralytics跟踪）
├── coordinate_tool.py       # 坐标标定工具
├── quick_start.py           # 快速入门示例
├── config_example.json      # 配置文件示例
├── README.md                # 本文档
├── WORKFLOW.md              # 详细工作流程指南
└── UPGRADE.md               # v2.0升级指南
```

## 核心算法

### 1. 目标跟踪

使用**Ultralytics内置跟踪器**，支持两种SOTA算法：

**ByteTrack**：
- 速度快，适合实时应用
- 利用低置信度检测框辅助跟踪
- 遮挡恢复能力强

**BoT-SORT**：
- 准确度最高
- 结合卡尔曼滤波和运动预测
- 适合复杂场景和快速移动目标

两者都远优于简单的IoU匹配，ID保持稳定，遮挡鲁棒性强。

### 2. 线段相交判定

使用参数方程判断轨迹段与绊线是否相交：
```python
# 两条线段: (p1, p2) 和 (p3, p4)
# 计算参数 t 和 u
# 当 0 ≤ t ≤ 1 且 0 ≤ u ≤ 1 时，线段相交
```

### 3. 方向判断

使用向量叉积判断穿越方向：
```python
# 绊线向量: v = p2 - p1
# 目标位置: prev (前一帧), curr (当前帧)
# 叉积: cross_prev = v × (prev - p1)
#       cross_curr = v × (curr - p1)
#
# cross_prev > 0: 目标在左侧
# cross_prev < 0: 目标在右侧
#
# 符号相反表示穿越:
#   cross_prev > 0 && cross_curr < 0 => 左到右
#   cross_prev < 0 && cross_curr > 0 => 右到左
```

## 配置文件格式

创建JSON配置文件定义绊线（参考 `config_example.json`）：

```json
{
  "tripwires": [
    {
      "id": "entrance_line",           // 绊线唯一标识
      "points": [[200, 400], [600, 350]],  // 线段起点和终点 [x, y]
      "direction": "left_to_right",    // 方向规则
      "enabled": true,                 // 是否启用
      "alert_cooldown": 2.0,           // 冷却时间（秒）
      "color": [0, 255, 0]             // 显示颜色 [B, G, R]
    }
  ]
}
```

### 方向选项

- `"left_to_right"`: 仅检测从左侧到右侧的穿越
- `"right_to_left"`: 仅检测从右侧到左侧的穿越
- `"bidirectional"`: 检测双向穿越

### 坐标系

- 原点(0, 0)在图像左上角
- X轴向右增长，Y轴向下增长

## 快速开始（两步工作流程）

绊线检测分为**两个独立步骤**：

```
步骤1: 坐标标定 → 生成配置文件 (tripwire_config.json)
              ↓
步骤2: 运行检测 → 使用配置文件进行实时检测
```

### 步骤1：标定绊线坐标

使用坐标标定工具在视频上标记绊线位置：

```bash
python tripwire_intrusion/coordinate_tool.py \
    --source your_video.mp4 \
    --output tripwire_config.json
```

**操作说明**（连续折线模式）：
1. **绘制连续折线**：
   - 第一条线段：点击两个点
   - 后续线段：只需点击一个点，自动从上一个点延续
   - 黄色圆圈标记下一条线段的起点
2. 按 `d` 设置整体方向（应用于所有线段：left_to_right / right_to_left / bidirectional）
3. 按 `s` 保存配置到JSON文件
4. 按 `r` 重置所有线段，按 `q` 退出

**适用场景**：适合绘制围栏、边界等连续区域，所有线段共享同一方向规则。

保存后会显示下一步命令和配置摘要。

### 步骤2：运行检测

使用生成的配置文件进行检测：

```bash
python tripwire_intrusion/tripwire_detector.py \
    --source your_video.mp4 \
    --weights your_model.pt \
    --config tripwire_config.json \
    --tracker bytetrack \
    --show --save
```

**详细工作流程请参考**: [WORKFLOW.md](WORKFLOW.md)

## 使用方法

### 1. 基本用法

```bash
# 处理视频文件（使用ByteTrack）
python tripwire_intrusion/tripwire_detector.py \
    --source video.mp4 \
    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
    --config tripwire_intrusion/config_example.json \
    --tracker bytetrack \
    --show --save

# 使用BoT-SORT（更准确，稍慢）
python tripwire_intrusion/tripwire_detector.py \
    --source video.mp4 \
    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
    --config tripwire_intrusion/config.json \
    --tracker botsort \
    --show --save

# 处理RTSP流
python tripwire_intrusion/tripwire_detector.py \
    --source rtsp://admin:password@192.168.1.100:554/stream \
    --weights your_model.pt \
    --config tripwire_intrusion/config.json \
    --tracker bytetrack \
    --show

# 处理USB摄像头
python tripwire_intrusion/tripwire_detector.py \
    --source 0 \
    --weights your_model.pt \
    --config tripwire_intrusion/config.json \
    --tracker bytetrack \
    --show
```

### 2. 完整参数

```bash
python tripwire_intrusion/tripwire_detector.py \
    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
    --device cuda:0 \
    --imgsz 640 \
    --conf-thresh 0.25 \
    --nms-iou 0.7 \
    --tracker bytetrack \
    --config tripwire_intrusion/config.json \
    --source video.mp4 \
    --output-dir runs/tripwire \
    --save \
    --show \
    --export-events \
    --draw-trajectory \
    --trajectory-length 30
```

### 3. 参数说明

#### 模型参数
- `--weights`: 模型权重文件路径（支持所有YOLO系列）
- `--device`: 计算设备 (cuda:0, cuda:1, cpu)
- `--imgsz`: 推理图像尺寸 (默认640)

#### 检测参数
- `--conf-thresh`: 置信度阈值 (0-1, 默认0.25)
- `--nms-iou`: NMS的IoU阈值 (0-1, 默认0.7)

#### 跟踪参数
- `--tracker`: 跟踪器类型
  - `bytetrack`: 更快，适合实时应用（推荐）
  - `botsort`: 更准确，适合复杂场景

#### 绊线配置
- `--config`: 绊线配置JSON文件路径

#### 输入输出
- `--source`: 输入源（视频路径/摄像头ID/RTSP地址）
- `--output-dir`: 输出目录 (默认 runs/tripwire)
- `--save`: 保存输出视频
- `--show`: 实时显示结果
- `--export-events`: 导出事件到JSON

#### 可视化
- `--draw-trajectory`: 绘制轨迹
- `--trajectory-length`: 轨迹显示长度 (默认30帧)

## 输出结果

### 1. 可视化视频

保存在 `--output-dir` 指定的目录，包含：
- 绊线标注（绿色线段，方向箭头）
- 检测框和目标ID
- 轨迹线（彩色，渐变透明度）
- 穿越事件提示（红色圆圈和箭头）
- 实时统计信息（活跃目标数、事件数、FPS）

### 2. 事件记录

使用 `--export-events` 导出JSON文件：

```json
{
  "total_events": 5,
  "events": [
    {
      "track_id": 3,
      "tripwire_id": "entrance_line",
      "direction": "left_to_right",
      "timestamp": 1701234567.89,
      "time_str": "2024-12-04 16:30:45",
      "position": [425.5, 380.2]
    }
  ]
}
```

## 性能优化

### 1. 提高检测速度

- 使用更小的模型（yolo11n/yolo11s）
- 降低推理尺寸（--imgsz 416 或 320）
- 使用GPU加速（--device cuda:0）
- 降低视频分辨率

### 2. 提高跟踪精度

- 使用 `--tracker botsort` 获得更高准确度（比bytetrack慢约10-20%）
- 使用更高的检测置信度（--conf-thresh 0.3-0.4）
- 降低 `--nms-iou` 减少框合并，保留更多检测

### 3. 减少误报

- 增加冷却时间（config中的 `alert_cooldown`）
- 设置合理的方向规则（避免双向检测噪声）
- 调整绊线位置，避开遮挡和边界区域

## 常见问题

### 1. 检测不到目标

- 检查模型权重和YAML是否匹配
- 降低 `--conf-thresh` 阈值
- 确认输入图像通道数与模型一致

### 2. 跟踪ID频繁切换

- 切换到 `--tracker botsort` 获得更稳定的ID
- 提高检测置信度（--conf-thresh 0.3）减少误检
- 检查视频质量和光照条件

### 3. 方向判断错误

- 检查绊线坐标顺序（起点到终点决定左右）
- 验证配置文件中的 `direction` 设置
- 使用可视化工具确认方向箭头

### 4. 重复触发报警

- 增加 `alert_cooldown` 冷却时间
- 检查轨迹是否在绊线附近徘徊

## 进阶功能

### 1. 多类别检测

修改 `tripwire_detector.py` 中的类别名称：

```python
self.class_names = {
    0: 'person',
    1: 'car',
    2: 'bicycle'
}
```

并在配置文件中添加类别过滤（需扩展代码）。

### 2. 区域预过滤

结合 `regional_intrusion` 模块，先用区域掩码过滤目标再进行绊线检测。

### 3. 报警回调

在 `TripwireMonitor.update()` 中添加回调函数：

```python
def alert_callback(event):
    """自定义报警处理"""
    print(f"🚨 报警: {event}")
    # 发送邮件、推送通知、触发继电器等

# 在 monitor.update() 中调用
if event:
    alert_callback(event)
```

## 依赖项

主要依赖已包含在项目的 `requirements.txt` 中：

```
torch>=1.12.1
opencv-python>=4.5.0
numpy>=1.21.0
ultralytics>=8.0.0
```

注意：v2.0不再需要scipy（已移除自定义跟踪器）。

## 相关文档

- **[WORKFLOW.md](WORKFLOW.md)** - 详细的两步工作流程指南（推荐先读）
- **[UPGRADE.md](UPGRADE.md)** - v1.0 → v2.0 升级指南
- **[config_example.json](config_example.json)** - 配置文件示例

## 引用

如果本系统对您的研究有帮助，请引用：

```bibtex
@article{yolov11-rgbt,
  title={YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework},
  author={...},
  journal={arXiv preprint arXiv:2506.14696},
  year={2025}
}
```

## 许可证

本项目基于 YOLOv11-RGBT，遵循其开源许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目地址]
- Email: [联系邮箱]

---

**更新日志**

- **v2.0** (2024-12-05):
  - ✅ 使用Ultralytics内置跟踪器（ByteTrack/BoT-SORT）
  - ✅ 跟踪准确度大幅提升
  - ✅ 优化坐标标定工具，支持指定输出路径
  - ✅ 完善两步工作流程（标定→检测）
  - ✅ 新增WORKFLOW.md详细指南

- **v1.0** (2024-12-04):
  - 实现基本绊线检测功能
  - 支持多输入源（视频/摄像头/RTSP）
  - 提供可视化和事件导出
