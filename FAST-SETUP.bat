@echo off
rem One-click setup of this fork: installs and builds on the first run, then tunes for a 32 GB PC and offers dual SSD streaming.
setlocal
set "PYTHONUTF8=1"
title SlenderEngine fast setup
cd /d "%~dp0"
if exist "strata-iq3_s.json" goto tune
echo.
echo  First run. This will:
echo    - download Qwen3.8-Flash-Next IQ3_S (allow about 120 GB of disk space)
echo    - build the engine from this folder, which needs the Visual Studio 2022 Build Tools
echo      with the C++ workload, the CUDA toolkit and CMake
echo.
choice /M " Carry on"
if errorlevel 2 exit /b 1
call START-HERE.bat --family qwen --model IQ3_S --build --context 262144 --kv int8 --vision gpu --no-start
if not exist "strata-iq3_s.json" (
  echo.
  echo  Setup did not finish, so nothing was tuned. See the messages above.
  pause
  exit /b 1
)
:tune
".venv\Scripts\python.exe" tools\fast_setup.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
