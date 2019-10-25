@echo off
ECHO Installing packages...

ECHO activating virtual environment
call venv\Scripts\activate

@echo on
python make_picks.py

PAUSE
