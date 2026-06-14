@echo off
echo ============================================
echo UAV-FAS 训练监控与报告生成
echo ============================================
echo.

:check_loop
echo [%date% %time%] 检查训练进度...

REM 检查训练目录中已完成的episode数
set "count=0"
for %%f in ("data\uav_bs_fas\scratch\td3_see_*\simulation_result_ep_*.mat") do (
    set /a count+=1
)

REM 简单方法：直接检查最新的td3_see_43目录
set "max_ep=0"
for %%f in ("data\storage\uav_bs_fas\scratch\td3_see_43\simulation_result_ep_*.mat") do (
    set "filename=%%~nf"
)

REM 使用python检查
D:\conda_envs\uav-fas\python.exe -c "import os,re; files=[f for f in os.listdir('data/storage/uav_bs_fas/scratch/td3_see_43') if f.startswith('simulation_result_ep_') and f.endswith('.mat')]; eps=[int(re.search(r'ep_(\d+)\.mat',f).group(1)) for f in files]; print(max(eps) if eps else 0)" > temp_ep_count.txt 2>nul
set /p MAX_EP=<temp_ep_count.txt

echo 当前最大episode: %MAX_EP%
echo 目标: 1000

if %MAX_EP% GEQ 1000 (
    echo.
    echo 训练已完成！开始生成报告...
    echo.
    cd /d "%~dp0"
    D:\conda_envs\uav-fas\python.exe run_and_report.py
    echo.
    echo 报告生成完成！
    goto :end
) else (
    echo 训练未完成，等待30秒后重新检查...
    timeout /t 30 /nobreak >nul
    goto :check_loop
)

:end
echo.
echo 按任意键退出...
pause >nul
