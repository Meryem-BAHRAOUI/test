@echo off
title Frontend - Port 8080
echo Starting Frontend (React + Vite) on port 8080...
cd /d "%~dp0frontend"
npm run dev
pause
