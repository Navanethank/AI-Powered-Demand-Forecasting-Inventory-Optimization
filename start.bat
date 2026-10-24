@echo off
title OptiSupply AI - Demand Forecasting & Inventory Optimization
cd /d "%~dp0"
echo ========================================================
echo  OptiSupply AI: Starting Web Dashboard...
echo ========================================================
echo Opening browser at http://127.0.0.1:5000
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5000
python run.py
pause
