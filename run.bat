@echo off
setlocal

echo VR Tracking Overlay
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if not exist "data\CameraRecord_20260505_165740.mp4" (
    echo.
    echo Missing file: data\CameraRecord_20260505_165740.mp4
    echo Put the source video into the data folder and run this file again.
    pause
    exit /b 1
)

if not exist "data\trackingData_20260505_165740.txt" (
    echo.
    echo Missing file: data\trackingData_20260505_165740.txt
    echo Put the tracking txt file into the data folder and run this file again.
    pause
    exit /b 1
)

echo.
echo Select rendering mode:
echo [1] First 10 seconds (fast test)
echo [2] Full video
echo =================================================
set /p choice="Enter option number (1 or 2): "
echo =================================================

if "%choice%"=="1" (
    echo.
    echo Rendering 10 seconds test video...
    ".venv\Scripts\python.exe" main.py --max-seconds 10 --output results\overlay_test_10s.mp4
    echo.
    echo Done!
    echo Video: results\overlay_test_10s.mp4
    echo Preview: results\preview.jpg
) else (
    echo.
    echo Rendering full video...
    ".venv\Scripts\python.exe" main.py --output results\overlay_full.mp4
    echo.
    echo Done!
    echo Video: results\overlay_full.mp4
    echo Preview: results\preview.jpg
)

pause