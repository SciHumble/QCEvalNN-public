@echo off
setlocal

echo === Activating virtual environment ===
call "%~dp0venv\Scripts\activate.bat"

echo === Running qc_eval.main as a module ===
cd /d "%~dp0"
python -m qc_eval.main

endlocal
pause

