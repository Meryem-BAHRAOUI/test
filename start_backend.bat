@echo off
title Backend - Port 8002
echo Starting Unified Backend (YOLOX + PaddleOCR + Ollama) on port 8002...
call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0backend"
python api_server.py
pause
