"""
UAV-FAS TD3 训练结果可视化报告生成器
中文版，按分类组织图片和讲解
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'data/storage/uav_bs_fas/scratch/td3_see_51'

print("=" * 60)
print("生成UAV-FAS训练可视化报告")
print("=" * 60)

# ==================== 1. 加载训练数据 ====================
print("\n[1] 加载训练数据...")

# 加载所有episode数据
episodes = []
episode_rewards = []
episode_capacities = []
episode_secure_caps = []
episode_uav_positions = []

ep_files = sorted([f for f in os.listdir(DATA_DIR)
                   if f.startswith('simulation_result_ep_') and f.endswith('.mat')],
                  key=lambda x: int(x.split('_ep_')[1].split('.')[0]))

for f in ep_files:
    ep_num = int(f.split('_ep_')[1].split('.')[0])
    data = loadmat(os.path.join(DATA_DIR, f))
    key = f'result_{ep_num}'
    if key in data:
        result = data[key]
        reward = result['reward'][0, 0].flatten()
        cap = result['user_capacity'][0, 0]
        secure = result['secure_capacity'][0, 0]
        uav = result['UAV_state'][0, 0]

        episodes.append(ep_num)
        episode_rewards.append(np.mean(reward))
        episode_capacities.append(np.mean(cap, axis=0))
        episode_secure_caps.append(np.mean(secure, axis=0))
        episode_uav_positions.append(uav[-1, :2])  # 最终位置

episodes = np.array(episodes)
episode_rewards = np.array(episode_rewards)
episode_capacities = np.array(episode_capacities)
episode_secure_caps = np.array(episode_secure_caps)
episode_uav_positions = np.array(episode_uav_positions)

print(f"  加载了 {len(episodes)} 个episode")
print(f"  平均奖励: {np.mean(episode_rewards):.4f}")
print(f"  平均容量: {np.mean(episode_capacities, axis=0)}")

# 加载轨迹数据
all_uav轨迹 = []
for f in ep_files[:5]:  # 取前5个episode的轨迹
    ep_num = int(f.split('_ep_')[1].split('.')[0])
    data = loadmat(os.path.join(DATA_DIR, f))
    key = f'result_{ep_num}'
    if key in data:
        result = data[key]
        uav = result['UAV_state'][0, 0]
        all_uav轨迹.append(uav[:, :2])

# ==================== 2. 加载系统配置 ====================
print("\n[2] 加载系统配置...")
from env_uav_bs_fas import MiniSystem

system = MiniSystem(
    user_num=2, bs_ant_num=8, fas_ant_num=12, ris_ant_num=64,
    if_dir_link=1, if_with_FAS=True, if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design='see', project_name='report_gen', step_num=100
)

# ==================== 3. 生成图表 ====================
print("\n[3] 生成图表...")

# ---------- 分类1: 系统拓扑图 ----------
print("  生成系统拓扑图...")
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
ax1.set_title('系统拓扑与UAV飞行轨迹', fontsize=16, fontweight='bold', pad=20)

# 绘制边界
border_x = [-25, 25, 25, -25, -25]
border_y = [0, 0, 50, 50, 0]
ax1.plot(border_x, border_y, 'k--', linewidth=2, alpha=0.5, label='区域边界')

# 绘制RIS覆盖范围 (示意)
ris_circle = plt.Circle((system.RIS.coordinate[0], system.RIS.coordinate[1]),
                        15, color='orange', alpha=0.1, label='RIS覆盖范围')
ax1.add_patch(ris_circle)

# 绘制UAV轨迹 (取几个episode)
colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(all_uav轨迹)))
for i, traj in enumerate(all_uav轨迹):
    ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.7)
    ax1.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=100, marker='o',
                edgecolors='black', linewidth=1, zorder=5)
    ax1.scatter(traj[-1, 0], traj[-1, 1], c=[colors[i]], s=100, marker='*',
                edgecolors='black', linewidth=1, zorder=5)

# 标记起点
ax1.scatter(-20, 0, c='blue', s=200, marker='s', edgecolors='black',
            linewidth=2, zorder=5, label='UAV起点 (-20, 0)')

# 标记用户
for i, user in enumerate(system.user_list):
    ax1.scatter(user.coordinate[0], user.coordinate[1], c='green', s=250, marker='D',
                edgecolors='black', linewidth=2, zorder=5, label=f'用户{i}')
    ax1.annotate(f'用户{i}\n({user.coordinate[0]:.0f}, {user.coordinate[1]:.0f})',
                xy=(user.coordinate[0], user.coordinate[1]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 标记攻击者
ax1.scatter(system.attacker_list[0].coordinate[0], system.attacker_list[0].coordinate[1],
            c='red', s=250, marker='X', edgecolors='black', linewidth=2, zorder=5,
            label='攻击者')
ax1.annotate(f'攻击者\n({system.attacker_list[0].coordinate[0]:.0f}, {system.attacker_list[0].coordinate[1]:.0f})',
            xy=(system.attacker_list[0].coordinate[0], system.attacker_list[0].coordinate[1]),
            xytext=(15, -20), textcoords='offset points',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 标记RIS
ax1.scatter(system.RIS.coordinate[0], system.RIS.coordinate[1], c='orange', s=350, marker='s',
            edgecolors='black', linewidth=2, zorder=5, label='RIS (64单元)')
ax1.annotate(f'RIS 8×8\n({system.RIS.coordinate[0]:.0f}, {system.RIS.coordinate[1]:.0f})',
            xy=(system.RIS.coordinate[0], system.RIS.coordinate[1]),
            xytext=(15, 15), textcoords='offset points',
            fontsize=10, color='orange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 绘制信号路径示意 (虚线)
for user in system.user_list:
    ax1.plot([system.RIS.coordinate[0], user.coordinate[0]],
             [system.RIS.coordinate[1], user.coordinate[1]],
             'g--', linewidth=1.5, alpha=0.5)

ax1.plot([system.RIS.coordinate[0], system.attacker_list[0].coordinate[0]],
         [system.RIS.coordinate[1], system.attacker_list[0].coordinate[1]],
         'r--', linewidth=1.5, alpha=0.5)

ax1.set_xlabel('X (米)', fontsize=12)
ax1.set_ylabel('Y (米)', fontsize=12)
ax1.set_xlim(-30, 30)
ax1.set_ylim(-5, 55)
ax1.legend(loc='upper left', fontsize=10, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'fig1_topology.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ---------- 分类2: 训练曲线 ----------
print("  生成训练曲线...")
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('训练曲线分析', fontsize=16, fontweight='bold', y=0.98)

# 图1: 奖励曲线
ax = axes[0, 0]
def smooth(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

ax.plot(episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='原始奖励')
smoothed = smooth(episode_rewards)
ax.plot(range(len(smoothed)), smoothed, 'b-', linewidth=2, label='平滑奖励')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('训练轮次', fontsize=11)
ax.set_ylabel('奖励值', fontsize=11)
ax.set_title('训练奖励曲线', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 图2: 用户容量曲线
ax = axes[0, 1]
ax.plot(episode_capacities[:, 0], 'g-', linewidth=1.5, label='用户1容量')
ax.plot(episode_capacities[:, 1], 'r-', linewidth=1.5, label='用户2容量')
ax.fill_between(range(len(episodes)), episode_capacities[:, 0], alpha=0.2, color='green')
ax.fill_between(range(len(episodes)), episode_capacities[:, 1], alpha=0.2, color='red')
ax.set_xlabel('训练轮次', fontsize=11)
ax.set_ylabel('容量 (bps/Hz)', fontsize=11)
ax.set_title('用户信道容量变化', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 图3: 安全容量曲线
ax = axes[1, 0]
ax.plot(episode_secure_caps[:, 0], 'g-', linewidth=1.5, label='用户1安全容量')
ax.plot(episode_secure_caps[:, 1], 'r-', linewidth=1.5, label='用户2安全容量')
ax.fill_between(range(len(episodes)), episode_secure_caps[:, 0], alpha=0.3, color='green')
ax.fill_between(range(len(episodes)), episode_secure_caps[:, 1], alpha=0.3, color='red')
ax.set_xlabel('训练轮次', fontsize=11)
ax.set_ylabel('安全容量 (bps/Hz)', fontsize=11)
ax.set_title('安全容量变化', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 图4: 容量对比柱状图
ax = axes[1, 1]
categories = ['用户1\n总容量', '用户1\n安全容量', '用户2\n总容量', '用户2\n安全容量']
values = [np.mean(episode_capacities[:, 0]), np.mean(episode_secure_caps[:, 0]),
          np.mean(episode_capacities[:, 1]), np.mean(episode_secure_caps[:, 1])]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#8BC34A']
bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('容量 (bps/Hz)', fontsize=11)
ax.set_title('平均容量对比', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(DATA_DIR, 'fig2_training_curves.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ---------- 分类3: 信道分析 ----------
print("  生成信道分析图...")
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('信道与波束成形分析', fontsize=16, fontweight='bold', y=0.98)

# 图1: RIS相位分布
ax = axes[0, 0]
ris_phases = np.angle(np.asarray(system.RIS.Phi_signal).diagonal())
ris_amps = np.abs(np.asarray(system.RIS.Phi_signal).diagonal())
scatter = ax.scatter(range(len(ris_phases)), np.degrees(ris_phases),
                    c=ris_amps, cmap='viridis', s=80, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax, label='幅度')
ax.set_xlabel('RIS单元索引', fontsize=11)
ax.set_ylabel('相位 (度)', fontsize=11)
ax.set_title('RIS相位分布', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 图2: 波束成形功率
ax = axes[0, 1]
G_power = np.abs(np.asarray(system.UAV_FAS.G)) ** 2
x = np.arange(G_power.shape[0])
width = 0.35
ax.bar(x - width/2, G_power[:, 0], width, label='用户1', color='green', alpha=0.7)
ax.bar(x + width/2, G_power[:, 1], width, label='用户2', color='red', alpha=0.7)
ax.set_xlabel('天线索引', fontsize=11)
ax.set_ylabel('功率', fontsize=11)
ax.set_title('波束成形功率分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 图3: 信道增益分析
ax = axes[1, 0]
from env_uav_bs_fas import MiniSystem as MS
test_sys = MS(user_num=2, bs_ant_num=8, fas_ant_num=12, ris_ant_num=64,
              if_dir_link=1, if_with_FAS=True, if_move_users=True, if_movements=True,
              reverse_x_y=(False, False), if_UAV_pos_state=True,
              reward_design='see', project_name='ch_analysis', step_num=10)

positions = np.array([[-20, 0], [-10, 10], [0, 20], [12.5, 12.5], [4, 47], [25, 25]])
gains_user0 = []
gains_user1 = []
for pos in positions:
    test_sys.reset()
    test_sys.UAV_FAS.coordinate[0] = pos[0]
    test_sys.UAV_FAS.coordinate[1] = pos[1]
    test_sys.UAV_FAS.F = np.asmatrix(
        np.ones((16, 1), dtype=complex) / np.sqrt(16)) * np.sqrt(test_sys.power_factor)
    for h in test_sys.h_U_k + test_sys.h_U_p + [test_sys.h_UR] + test_sys.h_R_k + test_sys.h_R_p:
        h.update_CSI()
    test_sys.update_channel_capacity()
    gains_user0.append(test_sys.user_list[0].capacity)
    gains_user1.append(test_sys.user_list[1].capacity)

labels = ['(-20,0)', '(-10,10)', '(0,20)', '(12.5,12.5)', '(4,47)', '(25,25)']
x_pos = np.arange(len(labels))
ax.bar(x_pos - 0.2, gains_user0, 0.4, label='用户1', color='green', alpha=0.7)
ax.bar(x_pos + 0.2, gains_user1, 0.4, label='用户2', color='red', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9, rotation=15)
ax.set_xlabel('UAV位置', fontsize=11)
ax.set_ylabel('容量 (bps/Hz)', fontsize=11)
ax.set_title('不同位置的信道容量', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 图4: 奖励分布
ax = axes[1, 1]
ax.hist(episode_rewards, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2,
           label=f'均值: {np.mean(episode_rewards):.3f}')
ax.axvline(np.median(episode_rewards), color='green', linestyle='--', linewidth=2,
           label=f'中位数: {np.median(episode_rewards):.3f}')
ax.set_xlabel('奖励值', fontsize=11)
ax.set_ylabel('频次', fontsize=11)
ax.set_title('奖励分布直方图', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(DATA_DIR, 'fig3_channel_analysis.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ---------- 分类4: 系统参数 ----------
print("  生成系统参数图...")
fig4, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

params_text = """
╔══════════════════════════════════════════════════════════════════════════╗
║                        UAV-FAS 系统参数配置                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  【硬件配置】                                                            ║
║  ├── 基站天线: 8根 (ULA均匀线性阵列)                                    ║
║  ├── 流体天线端口: 16个 (离散端口)                                      ║
║  ├── RIS反射单元: 64个 (8×8 UPA均匀面阵)                               ║
║  ├── 用户数: 2                                                          ║
║  ├── 攻击者数: 1                                                        ║
║  └── 工作频率: 28 GHz (毫米波)                                          ║
║                                                                          ║
║  【系统布局】                                                            ║
║  ├── UAV起点: (-20, 0, 50) 米                                          ║
║  ├── 用户0: (4, 47, 0.0001) 米                                         ║
║  ├── 用户1: (25, 25, 0.0001) 米                                        ║
║  ├── 攻击者: (20, 0, 0.0001) 米                                        ║
║  └── RIS: (12.5, 12.5, 0) 米 (地面固定)                               ║
║                                                                          ║
║  【训练参数】                                                            ║
║  ├── 算法: Twin Delayed DDPG (TD3)                                     ║
║  ├── 奖励设计: SEE (安全能效)                                           ║
║  ├── Agent 1学习率: 0.0003 (BS+FAS波束成形)                            ║
║  ├── Agent 2学习率: 0.0005 (UAV轨迹)                                   ║
║  ├── 批大小: 256 (Agent1), 128 (Agent2)                                ║
║  ├── Soft update: τ=0.005                                               ║
║  └── 训练轮次: 100 episodes × 100 steps                                ║
║                                                                          ║
║  【信道模型】                                                            ║
║  ├── 直射路径: UAV → 用户/攻击者 (LoS)                                 ║
║  ├── RIS反射路径: UAV → RIS → 用户/攻击者                              ║
║  ├── 路径损耗指数: n=3.5 (UAV-用户), n=2.2 (UAV-RIS)                  ║
║  └── 噪声功率: -114 dBm                                                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.05, 0.95, params_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'fig4_system_params.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("  所有图表已生成!")

# ==================== 4. 生成HTML报告 ====================
print("\n[4] 生成HTML报告...")

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UAV-FAS TD3 训练可视化报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.8; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}

        /* 标题样式 */
        h1 {{ text-align: center; color: #38bdf8; margin: 40px 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        h2 {{ color: #38bdf8; border-bottom: 3px solid #1e3a5f; padding-bottom: 10px; margin: 40px 0 20px; font-size: 1.8em; }}
        h3 {{ color: #7dd3fc; margin: 20px 0 10px; font-size: 1.3em; }}

        /* 卡片样式 */
        .card {{ background: #1e293b; border-radius: 16px; padding: 30px; margin: 20px 0; border: 1px solid #334155; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}

        /* 统计卡片 */
        .stat-card {{ background: linear-gradient(135deg, #1e3a5f, #1e293b); border-radius: 12px; padding: 25px; text-align: center; border: 1px solid #2563eb33; transition: transform 0.3s; }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; color: #38bdf8; }}
        .stat-label {{ color: #94a3b8; font-size: 0.95em; margin-top: 8px; }}

        /* 图表容器 */
        .chart-section {{ background: #0f172a; border-radius: 12px; padding: 20px; margin: 20px 0; }}
        .chart-title {{ color: #38bdf8; font-size: 1.2em; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #1e3a5f; }}
        .chart-desc {{ color: #94a3b8; font-size: 0.95em; margin-bottom: 15px; line-height: 1.6; }}
        img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }}

        /* 讲解样式 */
        .explanation {{ background: #1e3a5f33; border-left: 4px solid #38bdf8; padding: 20px; margin: 15px 0; border-radius: 0 12px 12px 0; }}
        .explanation-title {{ color: #38bdf8; font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }}
        .explanation p {{ color: #cbd5e1; line-height: 1.8; }}

        /* 指标样式 */
        .metric {{ display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px dashed #334155; }}
        .metric-label {{ color: #94a3b8; font-size: 0.95em; }}
        .metric-value {{ color: #38bdf8; font-weight: 600; font-size: 1.05em; }}

        /* 表格样式 */
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 14px 18px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #1e3a5f; color: #7dd3fc; font-weight: 600; }}
        tr:hover {{ background: #1e3a5f44; }}

        /* 代码样式 */
        .code {{ background: #0f172a; padding: 20px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 0.9em; overflow-x: auto; border: 1px solid #334155; white-space: pre-wrap; }}

        /* 徽章样式 */
        .badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 0.85em; font-weight: 600; margin: 3px; }}
        .badge-success {{ background: #065f46; color: #34d399; }}
        .badge-warning {{ background: #78350f; color: #fbbf24; }}
        .badge-info {{ background: #1e3a5f; color: #38bdf8; }}

        /* 分类标签 */
        .category-tag {{ display: inline-block; padding: 8px 20px; border-radius: 25px; font-size: 0.9em; font-weight: 600; margin: 5px; }}
        .tag-topology {{ background: #065f46; color: #34d399; }}
        .tag-training {{ background: #1e3a5f; color: #38bdf8; }}
        .tag-channel {{ background: #78350f; color: #fbbf24; }}
        .tag-params {{ background: #4c1d95; color: #c4b5fd; }}

        footer {{ text-align: center; padding: 40px; color: #64748b; font-size: 0.9em; border-top: 1px solid #334155; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UAV-FAS TD3 训练可视化报告</h1>
        <p style="text-align:center; color:#94a3b8; font-size:1.1em; margin-bottom:40px;">
            流体天线辅助无人机保密通信系统 | 强化学习训练结果分析
        </p>

        <!-- ==================== 训练概况 ==================== -->
        <div class="grid">
            <div class="stat-card">
                <div class="stat-value">{len(episodes)}</div>
                <div class="stat-label">训练轮次</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{np.mean(episode_rewards):.3f}</div>
                <div class="stat-label">平均奖励</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#34d399;">{np.mean(episode_capacities[:, 0]):.4f}</div>
                <div class="stat-label">用户1平均容量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#fbbf24;">{np.mean(episode_secure_caps[:, 0]):.4f}</div>
                <div class="stat-label">用户1平均安全容量</div>
            </div>
        </div>

        <!-- ==================== 分类1: 系统拓扑 ==================== -->
        <div class="card">
            <h2><span class="category-tag tag-topology">分类1</span> 系统拓扑与飞行轨迹</h2>

            <div class="chart-section">
                <div class="chart-title">系统拓扑与UAV飞行轨迹</div>
                <div class="chart-desc">
                    展示UAV-FAS系统的整体布局，包括UAV起点、用户位置、攻击者位置和RIS部署位置。
                    虚线表示RIS反射路径，彩色曲线表示不同训练轮次的UAV飞行轨迹。
                </div>
                <img src="fig1_topology.png" alt="系统拓扑图">
            </div>

            <div class="explanation">
                <div class="explanation-title">图表讲解</div>
                <p>
                    <strong>系统布局：</strong>UAV从(-20, 0)出发，需要飞向用户区域(右上方)同时避开攻击者(20, 0)。
                    RIS部署在(12.5, 12.5)，作为信号反射中继站。<br><br>
                    <strong>飞行轨迹：</strong>蓝色曲线展示了UAV在训练过程中的飞行路径。随着训练进行，UAV逐渐学会飞向最优位置，
                    以最大化用户容量同时最小化窃听者容量。<br><br>
                    <strong>信号路径：</strong>绿色虚线表示RIS到用户的反射路径，红色虚线表示RIS到攻击者的路径。
                    理想情况下，RIS应增强用户信号同时干扰攻击者。
                </p>
            </div>

            <div class="grid">
                <div>
                    <h3>系统节点位置</h3>
                    <table>
                        <tr><th>节点</th><th>位置 (x, y, z)</th><th>说明</th></tr>
                        <tr><td>UAV起点</td><td>(-20, 0, 50)</td><td>无人机初始位置</td></tr>
                        <tr><td>用户0</td><td>(4, 47, 0.0001)</td><td>地面用户</td></tr>
                        <tr><td>用户1</td><td>(25, 25, 0.0001)</td><td>地面用户</td></tr>
                        <tr><td>攻击者</td><td>(20, 0, 0.0001)</td><td>窃听者</td></tr>
                        <tr><td>RIS</td><td>(12.5, 12.5, 0)</td><td>智能反射面</td></tr>
                    </table>
                </div>
                <div>
                    <h3>信号传输模型</h3>
                    <div class="code">
直射路径: UAV → 用户/攻击者
RIS反射: UAV → RIS → 用户 (增强)
         UAV → RIS → 攻击者 (干扰)

等效信道: H_eff = h_direct + h_RIS_reflected
安全容量: C_s = max(0, R_user - R_attacker)
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== 分类2: 训练曲线 ==================== -->
        <div class="card">
            <h2><span class="category-tag tag-training">分类2</span> 训练曲线分析</h2>

            <div class="chart-section">
                <div class="chart-title">训练曲线综合分析</div>
                <div class="chart-desc">
                    展示TD3强化学习算法在100轮训练过程中的性能变化，包括奖励曲线、用户容量、安全容量和容量对比。
                </div>
                <img src="fig2_training_curves.png" alt="训练曲线">
            </div>

            <div class="explanation">
                <div class="explanation-title">图表讲解</div>
                <p>
                    <strong>左上 - 训练奖励曲线：</strong>蓝色细线为原始奖励，粗线为平滑后的趋势。
                    奖励值反映了系统整体性能，包括安全容量、功率约束和位置引导。<br><br>
                    <strong>右上 - 用户信道容量：</strong>绿色为用户1，红色为用户2。容量随训练变化反映了
                    波束成形优化效果。<br><br>
                    <strong>左下 - 安全容量变化：</strong>安全容量 = 用户容量 - 窃听者容量。
                    这是保密通信的核心指标，越高表示安全性越好。<br><br>
                    <strong>右下 - 容量对比柱状图：</strong>直观展示各指标的平均水平。
                </p>
            </div>

            <div class="grid">
                <div>
                    <h3>训练指标统计</h3>
                    <div class="metric">
                        <span class="metric-label">平均奖励</span>
                        <span class="metric-value">{np.mean(episode_rewards):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">奖励标准差</span>
                        <span class="metric-value">{np.std(episode_rewards):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">最终10轮平均奖励</span>
                        <span class="metric-value">{np.mean(episode_rewards[-10:]):.4f}</span>
                    </div>
                </div>
                <div>
                    <h3>容量指标统计</h3>
                    <div class="metric">
                        <span class="metric-label">用户1平均容量</span>
                        <span class="metric-value">{np.mean(episode_capacities[:, 0]):.6f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">用户1平均安全容量</span>
                        <span class="metric-value">{np.mean(episode_secure_caps[:, 0]):.6f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">安全容量占比</span>
                        <span class="metric-value">{np.mean(episode_secure_caps[:, 0])/(np.mean(episode_capacities[:, 0])+1e-10)*100:.1f}%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== 分类3: 信道分析 ==================== -->
        <div class="card">
            <h2><span class="category-tag tag-channel">分类3</span> 信道与波束成形分析</h2>

            <div class="chart-section">
                <div class="chart-title">信道特性与波束成形分析</div>
                <div class="chart-desc">
                    深入分析RIS相位分布、波束成形功率分配、不同位置的信道容量以及奖励分布特性。
                </div>
                <img src="fig3_channel_analysis.png" alt="信道分析">
            </div>

            <div class="explanation">
                <div class="explanation-title">图表讲解</div>
                <p>
                    <strong>左上 - RIS相位分布：</strong>64个RIS反射单元的相位和幅度分布。
                    颜色表示幅度大小，位置表示相位角度。优化的相位分布能实现信号的相长/相消干涉。<br><br>
                    <strong>右上 - 波束成形功率：</strong>8根基站天线对两个用户的功率分配。
                    绿色为用户1，红色为用户2。功率分配反映了波束成形的优化策略。<br><br>
                    <strong>左下 - 不同位置容量：</strong>展示UAV在不同位置时的用户容量。
                    这帮助理解为什么UAV需要飞向特定位置。RIS附近(12.5,12.5)通常有较好的反射增益。<br><br>
                    <strong>右下 - 奖励分布：</strong>训练过程中奖励值的分布情况。
                    红线为均值，绿线为中位数。
                </p>
            </div>

            <div class="grid">
                <div>
                    <h3>RIS配置</h3>
                    <table>
                        <tr><th>参数</th><th>值</th></tr>
                        <tr><td>反射单元数</td><td>64 (8×8 UPA)</td></tr>
                        <tr><td>单元间距</td><td>0.5λ ≈ 5.36mm</td></tr>
                        <tr><td>法线方向</td><td>[0, 1, 0] (指向y轴)</td></tr>
                        <tr><td>反射矩阵</td><td>64×64对角矩阵</td></tr>
                    </table>
                </div>
                <div>
                    <h3>波束成形配置</h3>
                    <table>
                        <tr><th>参数</th><th>值</th></tr>
                        <tr><td>BS天线数</td><td>8 (ULA)</td></tr>
                        <tr><td>FAS端口数</td><td>16 (离散)</td></tr>
                        <tr><td>功率约束</td><td>G_Pmax = 1600</td></tr>
                        <tr><td>功率因子</td><td>100</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- ==================== 分类4: 系统参数 ==================== -->
        <div class="card">
            <h2><span class="category-tag tag-params">分类4</span> 系统参数配置</h2>

            <div class="chart-section">
                <div class="chart-title">系统参数详细配置</div>
                <div class="chart-desc">
                    完整展示UAV-FAS系统的硬件配置、训练参数和信道模型参数。
                </div>
                <img src="fig4_system_params.png" alt="系统参数">
            </div>

            <div class="explanation">
                <div class="explanation-title">参数说明</div>
                <p>
                    <strong>硬件配置：</strong>系统采用8根BS天线+16端口FAS+64单元RIS的组合架构。
                    BS负责主动波束成形，FAS通过端口切换改变波束指向，RIS通过相位调整实现信号反射优化。<br><br>
                    <strong>训练参数：</strong>采用TD3算法，Agent 1控制BS+FAS波束成形(192维动作)，
                    Agent 2控制UAV轨迹(2维动作)。学习率经过调优以适应不同动作空间。<br><br>
                    <strong>信道模型：</strong>基于毫米波28GHz频段，采用双路径模型(直射+RIS反射)。
                    路径损耗遵循3GPP UMi模型。
                </p>
            </div>
        </div>

        <footer>
            <p>生成时间: 2026-05-28 | UAV-FAS TD3 训练可视化系统</p>
            <p>强化学习用于保密能源 — 流体天线辅助无人机保密通信</p>
        </footer>
    </div>
</body>
</html>"""

html_path = os.path.join(DATA_DIR, 'training_report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"  HTML报告已保存: {html_path}")

print("\n" + "=" * 60)
print("报告生成完成!")
print("=" * 60)
print(f"\n文件位置:")
print(f"  {DATA_DIR}/")
print(f"  ├── fig1_topology.png      (系统拓扑图)")
print(f"  ├── fig2_training_curves.png (训练曲线)")
print(f"  ├── fig3_channel_analysis.png (信道分析)")
print(f"  ├── fig4_system_params.png   (系统参数)")
print(f"  └── training_report.html     (完整报告)")
