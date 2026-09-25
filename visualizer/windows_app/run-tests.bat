@echo off
setlocal EnableDelayedExpansion
rem Isolate UI tests from a running gait sender on the production port.
set SNAKE_VISUALIZER_PORT=18781
cd /d "%~dp0"
call build-tests.bat
if errorlevel 1 exit /b 1
set "SNAKE_TEST_LOGS=%~dp0..\artifacts\windows-native-acceptance"
if not exist "%SNAKE_TEST_LOGS%" mkdir "%SNAKE_TEST_LOGS%"
for %%T in (CoreModelTests PaperGeometryTests CommunicationTests FeatureTests ProtocolPacketTests CameraReferenceTests UiSmokeTests SoakTests) do (
    "bin\%%T.exe" 15 > "%SNAKE_TEST_LOGS%\%%T.log" 2>&1
    if not "!errorlevel!"=="0" (
        type "%SNAKE_TEST_LOGS%\%%T.log"
        exit /b 1
    )
    type "%SNAKE_TEST_LOGS%\%%T.log"
)
echo All tests passed. No real servo bus was accessed.
