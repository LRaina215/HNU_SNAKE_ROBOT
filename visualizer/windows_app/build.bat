@echo off
setlocal
cd /d "%~dp0"
if not exist "bin" mkdir "bin"
set OUTPUT=bin\SnakeRobotVisualizer.exe
if not "%~1"=="" set OUTPUT=%~1
"%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:winexe /optimize+ /codepage:65001 /out:"%OUTPUT%" /reference:System.dll /reference:System.Core.dll /reference:System.Drawing.dll /reference:System.Windows.Forms.dll /reference:System.Web.Extensions.dll Program.cs Models.cs ProfileStore.cs FeetechProtocol.cs LocalControlServer.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs RuntimeEngine.cs Communication.cs ExperimentData.cs SnakeCanvas.cs ProfileEditorForm.cs MainFormV2.cs
if errorlevel 1 exit /b 1
echo Build complete: %OUTPUT%
