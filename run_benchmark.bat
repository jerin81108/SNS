@echo off
title Q-Signal Benchmark Runner
echo ======================================================
echo       Running Q-Signal Benchmark Experiment...
echo ======================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" experiments/run_benchmark.py --duration 120 --seeds 3 --verbose
) else (
    python experiments/run_benchmark.py --duration 120 --seeds 3 --verbose
)

echo.
echo Benchmark finished! Results saved in data/experiments/
pause
