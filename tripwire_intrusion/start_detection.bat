@echo off
chcp 65001 >nul 2>&1
cd /d %~dp0\..
REM Tripwire Intrusion Detection System - Quick Start Script (Windows)

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
set CONFIG_UPDATE_INTERVAL=30
set TRACKER=bytetrack
set DRAW_TRAJECTORY=true
set TRAJECTORY_LENGTH=30

REM ==================== Start Detection System ====================

echo ==========================================
echo Tripwire Intrusion Detection System
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
echo   Config Update Interval: %CONFIG_UPDATE_INTERVAL% seconds
echo   Tracker: %TRACKER%
echo   Draw Trajectory: %DRAW_TRAJECTORY%
echo   Trajectory Length: %TRAJECTORY_LENGTH%
echo.
echo Starting...
echo ==========================================
echo.

python tripwire_intrusion/tripwire_detector_api.py ^
    --api-url %API_URL% ^
    --username %USERNAME% ^
    --password %PASSWORD% ^
    --model-yaml %MODEL_YAML% ^
    --weights %WEIGHTS% ^
    --device %DEVICE% ^
    --target-size %TARGET_SIZE% ^
    --process-fps %PROCESS_FPS% ^
    --config-update-interval %CONFIG_UPDATE_INTERVAL% ^
    --tracker %TRACKER% ^
    --draw-trajectory %DRAW_TRAJECTORY% ^
    --trajectory-length %TRAJECTORY_LENGTH%

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
    echo 5. Check error logs above
    echo.
    pause
    exit /b 1
)
