@echo off
title Q-Signal Unit Tests
echo ======================================================
echo           Running Q-Signal Test Suite...
echo ======================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\pytest.exe" (
    ".venv\Scripts\pytest.exe" tests/ -v
) else (
    pytest tests/ -v
)

echo.
echo ======================================================
pause
