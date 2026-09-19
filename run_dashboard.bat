@echo off
title Q-Signal Traffic Optimization Dashboard
echo ======================================================
echo           Starting Q-Signal Dashboard...
echo ======================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    echo [OK] Virtual environment found.
    echo Launching Streamlit dashboard at http://localhost:8501 ...
    start "" http://localhost:8501
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo [!] Virtual environment not found in .venv!
    echo Trying system python...
    start "" http://localhost:8501
    python -m streamlit run app.py
)

pause
