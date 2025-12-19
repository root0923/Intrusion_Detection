# 20路并发压测工具

## 概述
用于测试系统在20路并发场景下的性能表现。所有路都指向同一个RTSP流，通过多进程模拟真实的多摄像头场景。

## 快速开始

### 1. 基础使用（推荐）
```bash
cd /Users/root1/Documents/jiuzhou/Intrusion_Detection
chmod +x unified_detector/test_tools/run_stress_test.sh
./unified_detector/test_tools/run_stress_test.sh
```

这将启动：
- 20路并发推理
- GPU性能监控
- 测试时长：5分钟

### 2. 自定义参数
```bash
# 测试10路，运行1分钟
./unified_detector/test_tools/run_stress_test.sh -n 10 -d 60

# 测试5路，运行30秒
./unified_detector/test_tools/run_stress_test.sh -n 5 -d 30
```

### 3. 分别运行（高级）

#### 只运行压测（不监控）
```bash
python unified_detector/test_tools/stress_test_main.py \
    --num-streams 20 \
    --duration 300
```

#### 只运行GPU监控
```bash
python unified_detector/test_tools/monitor_gpu.py \
    --interval 2 \
    --duration 300
```

## 工作原理

### 压测流程
1. 从后端API获取真实配置（你已部署的3个规则）
2. 复制20份，修改camera_key使其唯一
3. 每个进程独立运行，但都拉取同一个RTSP流
4. 统计每个进程的推理延迟、规则处理延迟

### 监控内容
- **GPU利用率**：实时GPU使用百分比
- **GPU显存**：已用/总显存（MB）
- **GPU温度**：摄氏度
- **CPU利用率**：系统CPU使用率
- **系统内存**：已用/总内存（GB）

## 输出文件

所有输出文件在：`unified_detector/test_tools/logs/`

### 1. 压测日志
- 文件名：`stress_test_YYYYMMDD_HHMMSS.log`
- 内容：
  - 每个进程的启动信息
  - 每5秒输出性能统计（推理延迟、规则延迟）
  - 进程存活状态

示例：
```
[camera_001] 帧: 150 | 推理: 45.2ms | 规则: 2.3ms | 总计: 47.5ms | 活跃规则: ['area_intrusion', 'tripwire_intrusion', 'water_safety']
```

### 2. GPU监控CSV
- 文件名：`gpu_monitor_YYYYMMDD_HHMMSS.csv`
- 内容：每2秒记录一次GPU/CPU/内存数据
- 可用Excel或Python分析

示例CSV：
```csv
timestamp,gpu_util_%,gpu_mem_used_mb,gpu_mem_total_mb,gpu_mem_%,gpu_temp_c,cpu_%,sys_mem_used_gb,sys_mem_total_gb,sys_mem_%
2025-12-19 16:30:00,85.0,12000,16384,73.2,68,45.2,8.5,16.0,53.1
```

## 关键指标解读

### GPU显存（RTX 5070 Ti有16GB）
- ✅ **<12GB**: 安全，还有余量
- ⚠️ **12-15GB**: 接近上限，注意观察
- ❌ **>15GB**: 超载，可能OOM

### 推理延迟
- ✅ **<50ms**: 优秀，可支持20路
- ⚠️ **50-100ms**: 可接受，但接近瓶颈
- ❌ **>100ms**: 延迟过高，无法满足实时性

### GPU利用率
- **<50%**: GPU还有余量，可增加并发数
- **50-80%**: 良好负载
- **>90%**: GPU满负荷，已达上限

## 参数说明

### stress_test_main.py
```bash
--num-streams 20         # 并发流数量
--duration 300          # 测试时长（秒）
--process-fps 10.0      # 每秒处理帧数
--target-size 640       # YOLO输入尺寸
--device cuda:0         # GPU设备
```

### monitor_gpu.py
```bash
--interval 2            # 监控间隔（秒）
--duration 300          # 监控时长（秒），0=无限
--output xxx.csv        # 输出文件路径
```

## 常见问题

### Q1: 显存不足怎么办？
**方案1**: 减少并发数
```bash
./run_stress_test.sh -n 10  # 改为10路
```

**方案2**: 减小模型输入尺寸
```bash
python stress_test_main.py --target-size 480  # 从640改为480
```

**方案3**: 降低处理帧率
```bash
python stress_test_main.py --process-fps 5  # 从10降到5
```

### Q2: 如何快速测试？
```bash
# 5路，运行30秒
./run_stress_test.sh -n 5 -d 30
```

### Q3: 进程异常退出怎么办？
- 查看日志文件中的错误信息
- 检查GPU驱动、CUDA版本
- 确认RTSP流地址正确

### Q4: 如何分析CSV数据？
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv('logs/gpu_monitor_xxx.csv')

# 绘制显存占用曲线
plt.plot(df['timestamp'], df['gpu_mem_%'])
plt.xlabel('时间')
plt.ylabel('GPU显存占用(%)')
plt.show()
```

## 测试建议

### 渐进式测试
1. **1路** → 验证基本功能
2. **5路** → 初步压力测试
3. **10路** → 中等负载
4. **15路** → 高负载
5. **20路** → 极限测试

每个阶段运行1-2分钟，观察：
- GPU显存是否超限
- 推理延迟是否可接受
- 进程是否稳定

### 长时稳定性测试
找到合适的并发数后：
```bash
./run_stress_test.sh -n 15 -d 3600  # 15路运行1小时
```

检查：
- 内存是否泄漏（逐渐增长）
- 性能是否衰减
- 进程是否崩溃

## 结果示例

### 理想输出（20路可行）
```
[2025-12-19 16:30:00] GPU: 75% | 显存: 11500/16384MB (70.2%) | 温度: 65°C
[camera_001] 推理: 42.1ms | 规则: 2.1ms | 总计: 44.2ms
[camera_020] 推理: 43.5ms | 规则: 2.3ms | 总计: 45.8ms
运行时间: 60s / 300s | 进程状态: 20 存活, 0 异常
```

### 超载输出（需要优化）
```
[2025-12-19 16:30:00] GPU: 98% | 显存: 15800/16384MB (96.4%) | 温度: 82°C
[camera_001] 推理: 156.3ms | 规则: 4.5ms | 总计: 160.8ms
运行时间: 30s / 300s | 进程状态: 18 存活, 2 异常
```
→ 显存接近上限，延迟过高，建议减少并发数

## 支持

如有问题，请查看：
- 日志文件：`unified_detector/test_tools/logs/*.log`
- GPU监控：`unified_detector/test_tools/logs/*.csv`
