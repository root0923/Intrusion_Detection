#!/bin/bash
#
# 20路并发压测启动脚本
#
# 功能：
# - 同时启动压测程序和GPU监控
# - 支持自定义参数
#

# 默认参数
NUM_STREAMS=20
DURATION=300  # 5分钟
MONITOR_INTERVAL=2

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-streams)
            NUM_STREAMS="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -n, --num-streams NUM    并发流数量 (默认: 20)"
            echo "  -d, --duration SEC       测试时长（秒） (默认: 300)"
            echo "  -h, --help              显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                      # 运行20路，5分钟"
            echo "  $0 -n 10 -d 60         # 运行10路，1分钟"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$(dirname "$0")/../.."

echo "======================================"
echo "20路并发压测工具"
echo "======================================"
echo "并发数: $NUM_STREAMS 路"
echo "时长: $DURATION 秒"
echo "======================================"
echo ""

# 创建日志目录
LOG_DIR="unified_detector/test_tools/logs"
mkdir -p "$LOG_DIR"

# 启动GPU监控（后台运行）
echo "[1/2] 启动GPU监控..."
python unified_detector/test_tools/monitor_gpu.py \
    --interval $MONITOR_INTERVAL \
    --duration $DURATION &

MONITOR_PID=$!
echo "GPU监控进程 PID: $MONITOR_PID"
sleep 2

# 启动压测程序（前台运行）
echo ""
echo "[2/2] 启动压测程序..."
python unified_detector/test_tools/stress_test_main.py \
    --num-streams $NUM_STREAMS \
    --duration $DURATION

# 等待GPU监控结束
echo ""
echo "等待GPU监控进程结束..."
wait $MONITOR_PID

echo ""
echo "======================================"
echo "测试完成！"
echo "日志目录: $LOG_DIR"
echo "======================================"
