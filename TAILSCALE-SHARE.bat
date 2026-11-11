@echo off
rem Share the web app with your own Tailscale devices (HTTPS, tailnet only). See docs\TAILSCALE.md.
rem   TAILSCALE-SHARE.bat                                    what is set up now (changes nothing)
rem   TAILSCALE-SHARE.bat on --config path\to\strata-x.json  share, with an API key (asks first)
rem   TAILSCALE-SHARE.bat off                                stop sharing
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0tools\tailscale_share.py" %*
set "RC=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %RC%
