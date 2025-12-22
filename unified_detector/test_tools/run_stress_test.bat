@echo off
REM 20路并发压测启动脚本 (Windows版本)
REM
REM 功能：
REM - 同时启动压测程序和GPU监控
REM - 支持自定义参数
REM

setlocal

REM 默认参数
set NUM_STREAMS=10
set DURATION=300
set MONITOR_INTERVAL=2

REM 解析命令行参数
:parse_args
if "%~1"=="" goto start_test
if /i "%~1"=="-n" (
    set NUM_STREAMS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--num-streams" (
    set NUM_STREAMS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-d" (
    set DURATION=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--duration" (
    set DURATION=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--help" goto show_help

echo 未知选项: %~1
echo 使用 -h 或 --help 查看帮助
exit /b 1

:show_help
echo 使用方法: %~nx0 [选项]
echo.
echo 选项:
echo   -n, --num-streams NUM    并发流数量 (默认: 20)
echo   -d, --duration SEC       测试时长（秒） (默认: 300)
echo   -h, --help              显示帮助信息
echo.
echo 示例:
echo   %~nx0                      # 运行20路，5分钟
echo   %~nx0 -n 10 -d 60         # 运行10路，1分钟
exit /b 0

:start_test
REM 切换到项目根目录
cd /d "%~dp0..\.."

echo ======================================
echo 20路并发压测工具
echo ======================================
echo 并发数: %NUM_STREAMS% 路
echo 时长: %DURATION% 秒
echo ======================================
echo.

REM 创建日志目录
set LOG_DIR=unified_detector\test_tools\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM 启动GPU监控（后台运行）
echo [1/2] 启动GPU监控...
start /B python unified_detector\test_tools\monitor_gpu.py --interval %MONITOR_INTERVAL% --duration %DURATION%
timeout /t 2 /nobreak >nul

REM 启动压测程序（前台运行）
echo.
echo [2/2] 启动压测程序...
python unified_detector\test_tools\stress_test_main.py --num-streams %NUM_STREAMS% --duration %DURATION%

echo.
echo ======================================
echo 测试完成！
echo 日志目录: %LOG_DIR%
echo ======================================

endlocal