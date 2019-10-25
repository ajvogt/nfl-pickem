@echo off
ECHO running pickem script...

ECHO activating virtual environment
call venv\Scripts\activate

@echo on
python make_picks.py

PAUSE
git