@echo off
chcp 65001 >nul 2>&1
cd /d %~dp0\..
REM Unified Detection Framework - Quick Start Script (Windows)

REM ==================== Configuration ====================

REM Backend API Configuration
set API_URL=http://localhost:8080
set USERNAME=admin
set PASSWORD=admin123

REM Model Configuration
set MODEL_YAML=ultralytics/cfg/models/11/yolo11x.yaml
set WEIGHTS=data/LLVIP_IF-yolo11x-e300-16-pretrained.pt
set DEVICE=cuda:0

REM Detection Configuration
set TARGET_SIZE=640
set PROCESS_FPS=10.0
set TRACKER=bytetrack

REM Update Configuration
set CONFIG_UPDATE_INTERVAL=30

REM Log Configuration
set LOG_DIR=unified_detector/log

REM ==================== Start Unified Detection System ====================

echo ==========================================
echo Unified Detection Framework
echo ==========================================
echo.
echo Configuration:
echo   API URL: %API_URL%
echo   Username: %USERNAME%
echo   Model: %MODEL_YAML%
echo   Weights: %WEIGHTS%
echo   Device: %DEVICE%
echo   Target Size: %TARGET_SIZE%
echo   Process FPS: %PROCESS_FPS%
echo   Tracker: %TRACKER%
echo   Config Update Interval: %CONFIG_UPDATE_INTERVAL% seconds
echo   Log Directory: %LOG_DIR%
echo.
echo Starting...
echo ==========================================
echo.

python unified_detector/main.py ^
    --api-url %API_URL% ^
    --username %USERNAME% ^
    --password %PASSWORD% ^
    --model-yaml %MODEL_YAML% ^
    --weights %WEIGHTS% ^
    --device %DEVICE% ^
    --target-size %TARGET_SIZE% ^
    --process-fps %PROCESS_FPS% ^
    --tracker %TRACKER% ^
    --config-update-interval %CONFIG_UPDATE_INTERVAL% ^
    --log-dir %LOG_DIR%

REM ==================== Error Handling ====================

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo Program exited with error!
    echo ==========================================
    echo.
    echo Troubleshooting:
    echo 1. Check if API URL is correct
    echo 2. Check if username/password is correct
    echo 3. Check if model files exist
    echo 4. Check CUDA environment (if using GPU)
    echo 5. Check error logs in: %LOG_DIR%
    echo.
    pause
    exit /b 1
)
