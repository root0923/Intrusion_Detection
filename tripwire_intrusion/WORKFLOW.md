# 绊线入侵检测 - 完整工作流程

本文档说明如何从零开始使用绊线入侵检测系统。

---

## 📋 概述

绊线检测分为**两个独立步骤**：

```
第一步: 坐标标定 → 生成配置文件 (tripwire_config.json)
                   ↓
第二步: 运行检测 → 使用配置文件进行实时检测
```

---

## 第一步：坐标标定 📍

### 目标
在视频上标定绊线位置，生成配置文件

### 命令

```bash
python tripwire_intrusion/coordinate_tool.py \
    --source your_video.mp4 \
    --output tripwire_config.json
```

### 参数说明

- `--source`: 视频文件路径（或摄像头ID：0, 1, ...）
- `--output`: 输出配置文件路径（默认：tripwire_config.json）

### 操作步骤

1. **启动工具**
   ```bash
   python tripwire_intrusion/coordinate_tool.py --source video.mp4
   ```

2. **绘制连续折线绊线**
   - 工具使用**连续折线模式**，适合绘制围栏、边界等连续区域
   - **第一条线段**: 点击两个点
   - **后续线段**: 只需点击一个点，自动从上一个点延续
   - 黄色圆圈标记下一条线段的起点
   - 所有线段形成一条连续的折线

   **示例**：
   ```
   点击顺序: P1 → P2 → P3 → P4
   生成线段:
     line_1: P1 → P2
     line_2: P2 → P3  (自动从P2开始)
     line_3: P3 → P4  (自动从P3开始)
   ```

3. **设置方向**（可选，应用于所有线段）
   - 按 `d` 键
   - 选择方向：
     - `1`: left_to_right（从左到右）
     - `2`: right_to_left（从右到左）
     - `3`: bidirectional（双向，默认）
   - 所有线段共享同一个方向设置

4. **保存配置**
   - 按 `s` 键
   - 输入保存路径（或按Enter使用默认路径）
   - 系统会显示配置摘要和下一步命令

5. **其他操作**
   - 按 `r` 重置所有线段，重新开始
   - 按 `q` 退出

### 输出示例

标定完成后，会生成类似这样的配置文件（所有线段共享同一方向）：

```json
{
  "tripwires": [
    {
      "id": "line_1",
      "points": [[200, 400], [600, 350]],
      "direction": "left_to_right",
      "enabled": true,
      "alert_cooldown": 2.0,
      "color": [0, 255, 0]
    },
    {
      "id": "line_2",
      "points": [[600, 350], [650, 250]],
      "direction": "left_to_right",
      "enabled": true,
      "alert_cooldown": 2.0,
      "color": [0, 255, 0]
    },
    {
      "id": "line_3",
      "points": [[650, 250], [700, 300]],
      "direction": "left_to_right",
      "enabled": true,
      "alert_cooldown": 2.0,
      "color": [0, 255, 0]
    }
  ]
}
```

注意：连续折线中，每条线段的起点是上一条的终点。

---

## 第二步：运行检测 🚀

### 目标
使用已保存的配置文件进行绊线入侵检测

### 基本命令

```bash
python tripwire_intrusion/tripwire_detector.py \
    --source your_video.mp4 \
    --weights your_model.pt \
    --config tripwire_config.json \
    --tracker bytetrack \
    --show --save
```

### 完整参数示例

```bash
python tripwire_intrusion/tripwire_detector.py \
    --source video.mp4 \
    --weights data/LLVIP_IF-yolo11x-e300-16-pretrained.pt \
    --config tripwire_config.json \
    --tracker bytetrack \
    --device cuda:0 \
    --imgsz 640 \
    --conf-thresh 0.25 \
    --nms-iou 0.7 \
    --output-dir runs/tripwire \
    --show \
    --save \
    --export-events \
    --draw-trajectory
```

### 参数说明

#### 必需参数
- `--source`: 输入视频路径（或RTSP流、摄像头ID）
- `--weights`: YOLO模型权重文件
- `--config`: 绊线配置文件（第一步生成的JSON文件）

#### 跟踪参数
- `--tracker`: 跟踪器类型
  - `bytetrack`（推荐）：快速，适合实时应用
  - `botsort`：更准确，适合复杂场景

#### 检测参数
- `--device`: 计算设备（cuda:0, cuda:1, cpu）
- `--imgsz`: 推理图像尺寸（默认640）
- `--conf-thresh`: 置信度阈值（默认0.25）
- `--nms-iou`: NMS IoU阈值（默认0.7）

#### 输出参数
- `--output-dir`: 输出目录（默认runs/tripwire）
- `--show`: 实时显示结果
- `--save`: 保存输出视频
- `--export-events`: 导出事件到JSON

#### 可视化参数
- `--draw-trajectory`: 绘制轨迹（默认开启）
- `--trajectory-length`: 轨迹显示长度（默认30帧）

### 输出结果

检测完成后，会在输出目录生成：

1. **可视化视频** - `<视频名>_tripwire.mp4`
   - 绘制了绊线、检测框、轨迹、事件提示

2. **事件记录** - `events.json`（如果使用了--export-events）
   ```json
   {
     "total_events": 5,
     "events": [
       {
         "track_id": 3,
         "tripwire_id": "line_1",
         "direction": "left_to_right",
         "timestamp": 1701234567.89,
         "time_str": "2024-12-04 16:30:45",
         "position": [425.5, 380.2]
       }
     ]
   }
   ```

---


## 🔧 高级配置

### 手动编辑配置文件

标定后，您可以手动编辑JSON配置文件来调整参数：

```json
{
  "tripwires": [
    {
      "id": "entrance",           // 修改ID名称
      "points": [[200, 400], [600, 350]],
      "direction": "left_to_right",
      "enabled": true,             // 禁用某条线设为false
      "alert_cooldown": 3.0,       // 调整冷却时间（秒）
      "color": [0, 255, 255]       // 修改显示颜色 [B, G, R]
    }
  ]
}
```

### 配置参数详解

| 参数 | 类型 | 说明 | 示例 |
|-----|------|-----|------|
| `id` | string | 绊线唯一标识 | "entrance_line" |
| `points` | array | 线段起点和终点坐标 | [[x1, y1], [x2, y2]] |
| `direction` | string | 方向规则 | "left_to_right" / "right_to_left" / "bidirectional" |
| `enabled` | boolean | 是否启用 | true / false |
| `alert_cooldown` | float | 冷却时间（秒） | 2.0 |
| `color` | array | BGR颜色 | [0, 255, 0]（绿色） |

### 方向判断规则

绊线方向由起点→终点决定：

```
        左侧
          ↑
    p1 -------→ p2
          ↓
        右侧

- left_to_right: 从上方区域穿越到下方区域
- right_to_left: 从下方区域穿越到上方区域
- bidirectional: 两个方向都检测
```

---

## 💡 最佳实践

### 1. 绊线位置选择

✅ **好的位置**：
- 人流必经之路
- 视野开阔，少遮挡
- 避开阴影边界
- 避开频繁晃动的物体

❌ **不好的位置**：
- 画面边缘（检测框可能不完整）
- 复杂背景（树叶、水面）
- 强光或阴影交界处

### 2. 方向设置建议

- **单向通道**：设置对应的单向检测（减少误报）
- **双向通道**：使用bidirectional
- **计数应用**：分别设置两条线（进/出）

### 3. 参数调优

**提高准确度**：
```bash
--tracker botsort         # 使用更准确的跟踪器
--conf-thresh 0.3         # 提高置信度阈值
```

**提高速度**：
```bash
--tracker bytetrack       # 使用快速跟踪器
--imgsz 416               # 降低推理尺寸
```

**减少误报**：
- 增加配置文件中的 `alert_cooldown`（如3.0秒）
- 调整绊线位置，避开干扰区域

---

## ❓ 常见问题

### Q1: 坐标标定时看不到视频首帧？
**A**: 确认视频路径正确，尝试使用绝对路径。

### Q2: 保存的配置文件在哪里？
**A**: 默认保存在当前目录的 `tripwire_config.json`，可以用 `--output` 指定路径。

### Q3: 可以修改已保存的配置吗？
**A**: 可以！直接用文本编辑器打开JSON文件修改，或重新运行标定工具。

### Q4: 检测时找不到配置文件？
**A**: 使用绝对路径，或确保相对路径正确：
```bash
--config D:/project/tripwire_config.json  # 绝对路径
--config ./tripwire_config.json            # 相对路径
```

### Q5: 如何验证配置是否正确？
**A**: 运行检测时加上 `--show` 参数，查看绊线是否显示在正确位置。

### Q6: 绊线方向判断不对？
**A**: 检查配置文件中的 `direction` 设置，或调整 `points` 的顺序（起点和终点对调）。

---

## 📚 相关文档

- [README.md](README.md) - 完整功能说明
- [UPGRADE.md](UPGRADE.md) - v2.0升级指南
- [config_example.json](config_example.json) - 配置文件示例

---

## 🆘 获取帮助

如有问题，请：
1. 查看 [README.md](README.md) 的常见问题部分
2. 提交 GitHub Issue
3. 查看项目文档

---

**祝您使用愉快！** 🎉
