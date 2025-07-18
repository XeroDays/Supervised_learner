@echo off
cd /d "%~dp0"

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the script
python compare.py

:: Keep the window open
pause
