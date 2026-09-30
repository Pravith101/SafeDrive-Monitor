@echo off
setlocal
cd /d "%~dp0"
python app.py
if errorlevel 1 (
  echo.
  echo SafeDrive Monitor stopped with an error. Review the message above.
  pause
)
