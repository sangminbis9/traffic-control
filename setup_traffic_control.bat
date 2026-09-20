@echo off
chcp 65001 >nul
title Traffic Control Lab Setup

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_traffic_control.ps1"
if errorlevel 1 (
    echo.
    echo Traffic Control Lab setup failed. Check the message above.
    pause
)
