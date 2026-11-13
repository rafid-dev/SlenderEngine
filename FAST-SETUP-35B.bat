@echo off
rem One-click setup of Qwen3.6-35B-A3B (Unsloth UD-Q4_K_M with its MTP block) on this fork: setup's install on the first
rem run (download, engine build for this card, tokenizer), then tools\fast_setup_q36.py writes a launcher with the switches
rem measured in docs\fork\QWEN36-12GB.md, chosen for this PC's card, RAM and CPU (run-qwen36-fast.bat).
setlocal
set "PYTHONUTF8=1"
title SlenderEngine fast setup: Qwen3.6-35B-A3B
cd /d "%~dp0"
if exist "strata-qwen36-ud-q4_k_m.json" goto tune
echo.
echo  First run. This will:
echo    - download Qwen3.6-35B-A3B UD-Q4_K_M with its MTP block (about 23 GB; allow about 45 GB of disk space with the
echo      engine's expert cache file, which is written next to it on the first start)
echo    - build the engine from this folder for your card, which needs the Visual Studio 2022 Build Tools with the C++
echo      workload, the CUDA toolkit and CMake
echo.
choice /M " Carry on"
if errorlevel 2 exit /b 1
call START-HERE.bat --family qwen36 --model UD-Q4_K_M --build --context 262144 --no-start
if not exist "strata-qwen36-ud-q4_k_m.json" (
  echo.
  echo  Setup did not finish, so nothing was tuned. See the messages above.
  pause
  exit /b 1
)
:tune
".venv\Scripts\python.exe" tools\fast_setup_q36.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
