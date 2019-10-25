:: installation shell script
@echo off
ECHO Installing packages...

:: ensuring virtualenv is installed
@echo on
pip3 install virtualenv
virtualenv venv

:: activate virtual environment and installing packages
call venv\Scripts\activate

@echo on
pip3 install -r requirements.txt

@echo off
echo install successful!
echo to activate environment, use "> venv\Scripts\activate"
echo to run script, "> python make_picks.py"
PAUSE
