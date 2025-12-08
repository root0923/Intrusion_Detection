# 区域入侵检测系统

基于 YOLOv11-RGBT 的红外视频区域入侵检测与报警系统。

## 功能特点

✅ **多ROI区域检测**：支持多个多边形ROI区域，使用mask方式只检测区域内目标
✅ **高效检测**：合并所有ROI为一个mask，只需检测一次
✅ **智能报警**：三级报警参数控制（置信度阈值、消抖时间、重复报警间隔）
✅ **报警接口**：支持HTTP POST接口报警通知
✅ **可视化**：实时显示检测结果、ROI区域和报警信息
✅ **报警记录**：自动保存报警截图和输出视频

## 核心原理

系统采用 **合并ROI Mask** 方式进行检测：
1. 将所有ROI区域合并为一个总mask
2. 将ROI区域外的像素变黑
3. 在masked图像上进行**一次**YOLO检测
4. 检测到的任何目标即为ROI区域内的入侵

**优势**：
- 无需对每个ROI单独检测，性能大幅提升
- 无需后处理判断目标是否在ROI内
- 检测器只关注ROI区域，提高检测精度
- 支持任意形状的多边形ROI

## 文件结构

```
regional_intrusion/
├── intrusion_detector.py       # 主程序：区域入侵检测系统
├── rois_config.json            # ROI配置文件（多边形顶点坐标）
├── visualize_roi_mask.py       # ROI可视化工具
└── README.md                   # 本文档
```

## 快速开始

### 1. 配置ROI区域

编辑 `rois_config.json`：

```json
{
  "image_width": 640,
  "image_height": 480,
  "rois": [
    [
      [x1, y1],
      [x2, y2],
      [x3, y3],
      ...
    ],
    [
      [x1, y1],
      [x2, y2],
      ...
    ]
  ]
}
```

**参数说明**：
- `image_width/image_height`：视频分辨率
- `rois`：ROI列表，每个ROI是多边形顶点坐标数组
- **支持多个ROI**，系统会自动合并为一个mask

### 2. 配置检测参数

在 `intrusion_detector.py` 的 `main()` 函数中修改：

```python
# 模型配置
YAML_PATH = "ultralytics/cfg/models/11/yolo11x.yaml"
WEIGHTS_PATH = "data/LLVIP_IF-yolo11x-e300-16-pretrained.pt"
DEVICE = 'cuda:0'

# ROI配置
ROI_CONFIG = "regional_intrusion/rois_config.json"

# 视频路径
VIDEO_PATH = "data/infrared_video.mp4"  # 输入视频
OUTPUT_PATH = "runs/intrusion_detection/output_video.mp4"  # 输出视频

# 报警参数
CONF_THRESHOLD = 0.5          # 置信度阈值
FIRST_ALARM_DURATION = 2.0    # 首次报警时间（秒）- 消抖
REPEAT_ALARM_INTERVAL = 30.0  # 重复报警间隔（秒）
ALARM_URL = None              # 报警接口URL
```

**报警参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `CONF_THRESHOLD` | 置信度阈值，只有高于此值的检测才触发报警 | 0.5 |
| `FIRST_ALARM_DURATION` | 首次报警时间（消抖），目标持续出现该时长后才触发首次报警 | 2.0秒 |
| `REPEAT_ALARM_INTERVAL` | 重复报警间隔，该时间段内只报警一次 | 30.0秒 |
| `ALARM_URL` | 报警HTTP接口URL，设置为None则不发送 | None |

### 3. 运行检测

```bash
cd regional_intrusion
python intrusion_detector.py
```

**运行时操作**：
- 按 `q` 键：退出程序
- 按 `s` 键：保存当前帧截图

### 4. 查看结果

输出内容：
- **输出视频**：`runs/intrusion_detection/output_video.mp4`
- **报警截图**：`runs/intrusion_detection/alarms/alarm_frame{n}.jpg`
- **终端日志**：实时显示检测和报警信息

## 报警接口

### 接口格式

系统通过 HTTP POST 发送报警信息到指定接口：

```python
ALARM_URL = "http://your-server.com/api/alarm"
```

### 报警数据格式

```json
{
  "timestamp": "2025-12-04 14:30:15",
  "frame_count": 1234,
  "detection_count": 2,
  "detections": [
    {
      "class": 0,
      "confidence": 0.87,
      "box": [120.5, 200.3, 180.2, 350.8]
    },
    {
      "class": 0,
      "confidence": 0.92,
      "box": [300.1, 180.5, 360.8, 400.2]
    }
  ]
}
```

**注意**：报警数据不再包含 `roi_id` 字段，因为系统不区分具体哪个ROI被入侵。

### 接口实现示例（Flask）

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/alarm', methods=['POST'])
def receive_alarm():
    alarm_data = request.json

    print(f"[报警] 时间: {alarm_data['timestamp']}")
    print(f"       检测数: {alarm_data['detection_count']}")

    # 这里可以添加您的报警处理逻辑：
    # - 发送邮件/短信通知
    # - 保存到数据库
    # - 触发其他系统联动

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## 工作流程

```
读取视频帧 → 应用合并ROI mask → YOLO检测(一次) → 判断是否有入侵
                                                      ↓
                                    有入侵 → 记录状态 → 持续>消抖时间? → 触发报警 → 发送接口
                                                      ↓
                                    无入侵 → 清除状态
```

## 类和方法说明

### ROIManager

ROI区域管理器，负责加载配置和应用mask。

**主要方法**：
- `__init__(config_path)`：加载ROI配置，创建合并mask
- `apply_mask(image)`：将所有ROI外区域变黑
- `draw_rois(image)`：绘制所有ROI区域可视化

### AlarmManager

报警管理器，负责报警逻辑和状态管理（简化版，不区分ROI）。

**主要方法**：
- `__init__(...)`：初始化报警参数
- `update_intrusion(detections, ...)`：更新入侵状态并触发报警
- `_send_alarm(alarm)`：发送报警到HTTP接口

### IntrusionDetectionSystem

入侵检测系统主类，整合检测器、ROI和报警功能。

**主要方法**：
- `__init__(detector, roi_config_path, ...)`：初始化系统
- `process_video(video_path, ...)`：处理视频并进行检测
- `_visualize(frame, detections, alarms)`：可视化检测结果

## 辅助工具

### ROI可视化工具

使用 `visualize_roi_mask.py` 预览ROI配置效果：

```bash
python visualize_roi_mask.py \
    --image data/dataset/infrared/n286.jpg \
    --config rois_config.json \
    --mode mask_outside
```

**参数说明**：
- `--mode mask_inside`：将ROI区域内变黑（ROI外保留）
- `--mode mask_outside`：将ROI区域外变黑（ROI内保留）⭐推荐用于检测

## 常见问题

### Q1: 如何标定ROI区域？

可以使用图像标注工具（如labelImg、CVAT）标注多边形，或编写简单的鼠标点击程序获取坐标。

### Q2: 视频分辨率与ROI配置不匹配怎么办？

系统会自动将ROI mask缩放到视频分辨率，但建议ROI配置与实际视频分辨率一致以获得最佳效果。

### Q3: 如何调整报警灵敏度？

调整三个参数：
- **降低灵敏度**：提高 `CONF_THRESHOLD`，增加 `FIRST_ALARM_DURATION`
- **提高灵敏度**：降低 `CONF_THRESHOLD`，减少 `FIRST_ALARM_DURATION`

### Q4: 多个ROI可以重叠吗？

可以重叠。系统会将所有ROI合并为一个mask，重叠区域会被包含在检测范围内。

### Q5: 为什么不区分具体哪个ROI被入侵？

**设计理念**：
- 简化逻辑，提高性能（只检测一次）
- 大多数场景下只需知道"是否有入侵"，不需要区分具体ROI
- 如果需要区分ROI，建议为每个ROI运行独立的检测实例

### Q6: 如何提高检测速度？

- 使用更小的YOLO模型（如yolo11n替代yolo11x）
- 降低视频分辨率
- 使用GPU加速（设置 `DEVICE='cuda:0'`）
- **相比per-ROI检测，合并mask方式已大幅提升速度**

## 技术细节

### ROI Mask实现

```python
# 创建合并所有ROI的mask
mask = np.zeros((height, width), dtype=np.uint8)
for roi in rois:
    cv2.fillPoly(mask, [roi], 255)

# 应用mask（将所有ROI外变黑）
masked_image = image.copy()
masked_image[mask == 0] = 0
```

### 报警状态机

系统维护一个全局入侵状态：
```python
intrusion_state = {
    'first_time': timestamp,      # 首次检测到入侵的时间
    'last_alarm_time': timestamp  # 最后一次报警的时间
}
```

**状态转换**：
1. **检测到目标** → 记录 `first_time`，进入消抖阶段
2. **持续检测且超过消抖时间** → 触发首次报警，记录 `last_alarm_time`
3. **持续检测且超过重复报警间隔** → 触发重复报警，更新 `last_alarm_time`
4. **未检测到目标** → 清除状态

## 性能指标

在测试环境下的性能表现：

| 配置 | FPS | 备注 |
|------|-----|------|
| YOLOv11x + RTX 3090 | ~45 | 合并mask，640分辨率 |
| YOLOv11n + RTX 3090 | ~120 | 合并mask，640分辨率 |

*注：使用合并mask方式，FPS不受ROI数量影响*

## 许可证

本项目继承 YOLOv11-RGBT 的许可证。

## 联系方式

如有问题或建议，请提交 Issue 到项目仓库。
