@echo off
setlocal
cd /d "%~dp0"
if not exist "windows_app\bin\SnakeRobotVisualizer.exe" (
    call windows_app\build.bat
    if errorlevel 1 exit /b 1
)
start "" "windows_app\bin\SnakeRobotVisualizer.exe"
