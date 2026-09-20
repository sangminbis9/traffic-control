@echo off
chcp 65001 >nul
title Traffic Control Lab Stopper

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_traffic_control.ps1"
if errorlevel 1 (
    echo.
    echo Failed to stop Traffic Control Lab. Check the message above.
    pause
)
