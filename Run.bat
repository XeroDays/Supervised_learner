@echo off
cd /d "%~dp0"

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the script
python start.py

:: Keep the window open
pause
