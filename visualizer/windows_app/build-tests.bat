@echo off
setlocal
cd /d "%~dp0"
if not exist bin mkdir bin
set CSC=%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\CoreModelTests.exe" /reference:System.dll /reference:System.Core.dll Models.cs FeetechProtocol.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs CoreModelTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\FeatureTests.exe" /reference:System.dll /reference:System.Core.dll /reference:System.Drawing.dll /reference:System.Windows.Forms.dll /reference:System.Web.Extensions.dll Models.cs ProfileStore.cs FeetechProtocol.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs RuntimeEngine.cs Communication.cs ExperimentData.cs SnakeCanvas.cs FeatureTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\ProtocolPacketTests.exe" /reference:System.dll /reference:System.Core.dll FeetechProtocol.cs ProtocolPacketTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\HardwareProbe.exe" /reference:System.dll /reference:System.Core.dll FeetechProtocol.cs HardwareProbe.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\CameraReferenceTests.exe" /reference:System.dll /reference:System.Core.dll /reference:System.Drawing.dll /reference:System.Windows.Forms.dll Models.cs FeetechProtocol.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs SnakeCanvas.cs CameraReferenceTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\UiSmokeTests.exe" /reference:System.dll /reference:System.Core.dll /reference:System.Drawing.dll /reference:System.Windows.Forms.dll /reference:System.Web.Extensions.dll Models.cs ProfileStore.cs FeetechProtocol.cs LocalControlServer.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs RuntimeEngine.cs Communication.cs ExperimentData.cs SnakeCanvas.cs ProfileEditorForm.cs MainFormV2.cs UiSmokeTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\SoakTests.exe" /reference:System.dll /reference:System.Core.dll /reference:System.Web.Extensions.dll Models.cs ProfileStore.cs FeetechProtocol.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs RuntimeEngine.cs Communication.cs ExperimentData.cs SoakTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\PaperGeometryTests.exe" /reference:System.dll /reference:System.Core.dll /reference:System.Web.Extensions.dll Models.cs FeetechProtocol.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs ExperimentData.cs PaperGeometryTests.cs
if errorlevel 1 exit /b 1

"%CSC%" /nologo /target:exe /optimize+ /codepage:65001 /out:"bin\CommunicationTests.exe" /reference:System.dll /reference:System.Core.dll Models.cs FeetechProtocol.cs Kinematics.cs RollingAxis.cs CrossSectionCurve.cs RuntimeEngine.cs Communication.cs CommunicationTests.cs
if errorlevel 1 exit /b 1

echo All test executables built.
