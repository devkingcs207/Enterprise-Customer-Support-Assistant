@echo off
title Enterprise Customer Support Assistant

cd /d "%~dp0"

echo Starting Customer Support Assistant...

call ".venv\Scripts\activate.bat"

python -m streamlit run src/interface/app.py

pause