@echo off
chcp 936 >nul
title 实时手势识别（0-5数字手势）
cd /d %~dp0

REM 如果项目虚拟环境不存在，则自动创建并安装依赖
if not exist ".venv\Scripts\python.exe" (
    echo 首次运行，正在创建虚拟环境并安装依赖，请耐心等待...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请先安装 Python 3.9 及以上版本
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple
)

REM 启动实时手势识别主程序
.venv\Scripts\python.exe main.py %*
pause