@echo off
chcp 936 >nul
title 手势识别项目自测
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [提示] 尚未创建虚拟环境，请先双击运行 run.bat 完成环境安装
    pause
    exit /b 1
)

.venv\Scripts\python.exe self_test.py
pause