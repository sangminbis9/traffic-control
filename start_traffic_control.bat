@echo off
chcp 65001 >nul
title Traffic Control Lab Launcher

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_traffic_control.ps1"
if errorlevel 1 (
    echo.
    echo Failed to start Traffic Control Lab. Check the message above.
    pause
)
