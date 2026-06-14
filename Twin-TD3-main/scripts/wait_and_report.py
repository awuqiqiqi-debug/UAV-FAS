"""
UAV-FAS 训练监控脚本
训练完成后自动生成中文HTML报告
"""
import os
import re
import time
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TARGET_EPISODES = 1000
TRAIN_DIR = 'data/storage/uav_bs_fas/scratch/td3_see_43'

def get_max_episode():
    """获取当前最大episode数"""
    if not os.path.exists(TRAIN_DIR):
        return 0

    mat_files = [f for f in os.listdir(TRAIN_DIR)
                if f.startswith('simulation_result_ep_') and f.endswith('.mat')]

    max_ep = 0
    for f in mat_files:
        match = re.search(r'ep_(\d+)\.mat', f)
        if match:
            ep_num = int(match.group(1))
            if ep_num > max_ep:
                max_ep = ep_num
    return max_ep

print("=" * 60)
print("UAV-FAS 训练监控与报告生成")
print("=" * 60)
print(f"目标: {TARGET_EPISODES} 轮")
print(f"训练目录: {TRAIN_DIR}")
print()

while True:
    current_ep = get_max_episode()
    progress = current_ep / TARGET_EPISODES * 100

    print(f"\r当前进度: {current_ep}/{TARGET_EPISODES} ({progress:.1f}%)", end="", flush=True)

    if current_ep >= TARGET_EPISODES:
        print(f"\n\n训练已完成！共 {current_ep} 轮")
        print("开始生成报告...")

        # 运行报告生成脚本
        result = subprocess.run(
            ["D:/conda_envs/uav-fas/python.exe", "run_and_report.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("\n报告生成成功！")
            print(result.stdout)
        else:
            print(f"\n报告生成失败: {result.stderr}")

        break

    # 每30秒检查一次
    time.sleep(30)
