@echo off
REM Create virtual environment named ".venv"
python -m venv .venv

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
pip install -r requirements.txt

echo Virtual environment created and dependencies installed!
pause
