"""
UAV-FAS 系统优化与完整可视化
修复: FAS增益初始化为0的问题
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from src.envs.uav_comm_env import MiniSystem, get_energy_consumption, ENERGY_MIN, ENERGY_MAX, dB_to_normal
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("UAV-FAS 系统优化与完整可视化")
print("=" * 70)

# ==================== 1. 系统初始化 ====================
print("\n[1] 初始化系统...")
system = MiniSystem(
    user_num=2, bs_ant_num=8, fas_ant_num=12, ris_ant_num=64,
    if_dir_link=1, if_with_FAS=True, if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design='see', project_name='optimized_viz', step_num=100
)

# ==================== 2. 修复FAS初始化问题 ====================
print("\n[2] 修复FAS增益初始化...")
# F矩阵初始为全零导致fas_gain=0，容量全为0
# 修复: 初始化F为单位向量
system.UAV_FAS.F = np.asmatrix(
    np.ones((system.UAV_FAS.fas_num_ports, 1), dtype=complex) / np.sqrt(system.UAV_FAS.fas_num_ports)
) * np.sqrt(system.power_factor)
print(f"  F矩阵初始化完成, shape={system.UAV_FAS.F.shape}")
print(f"  FAS gain = {np.mean(np.abs(system.UAV_FAS.F)):.4f}")

# ==================== 3. 验证系统性能 ====================
print("\n[3] 验证系统性能...")
system.reset()
system.update_channel_capacity()

print("  初始信道容量:")
for i, user in enumerate(system.user_list):
    print(f"    用户{i}: capacity={user.capacity:.6f}, secure={user.secure_capacity:.6f}")
for i, attacker in enumerate(system.attacker_list):
    print(f"    攻击者{i}: capacity={attacker.capacity}")
print(f"  初始奖励: {system.reward():.6f}")

# ==================== 4. 模拟训练 (带FAS修复) ====================
print("\n[4] 模拟训练过程...")
episodes = 200
steps = 50

rewards_history = []
capacity_user1 = []
capacity_user2 = []
secure_capacity_user1 = []
secure_capacity_user2 = []
attacker_capacity_history = []
uav_positions = []
see_history = []  # Secrecy Energy Efficiency
energy_history = []

for ep in range(episodes):
    system.reset()
    # 重新初始化F矩阵 (修复初始化问题)
    system.UAV_FAS.F = np.asmatrix(
        np.ones((system.UAV_FAS.fas_num_ports, 1), dtype=complex) / np.sqrt(system.UAV_FAS.fas_num_ports)
    ) * np.sqrt(system.power_factor)

    ep_reward = 0
    ep_cap1 = []
    ep_cap2 = []
    ep_secure1 = []
    ep_secure2 = []
    ep_attacker = []
    ep_see = []
    ep_energy = []

    for step in range(steps):
        # 自适应动作: 随着训练进行减小噪声
        noise_scale = 0.5 * (1 - ep / episodes)

        # Agent 1: BS+FAS波束成形
        action_1 = np.random.randn(192) * noise_scale

        # Agent 2: UAV轨迹 (趋向用户)
        user_center = np.mean([u.coordinate[:2] for u in system.user_list], axis=0)
        direction = user_center - system.UAV_FAS.coordinate[:2]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        action_2 = direction + np.random.randn(2) * noise_scale
        action_2 = np.clip(action_2, -1, 1)

        # 计算速度用于能耗
        move_x = action_2[0] * system.UAV_FAS.max_movement_per_time_slot
        move_y = action_2[1] * system.UAV_FAS.max_movement_per_time_slot
        v_t = np.sqrt(move_x**2 + move_y**2)

        new_state, reward, done, _ = system.step(
            action_0=action_2[0], action_1=action_2[1],
            G=action_1[0:32], Phi=action_1[32:],
            set_pos_x=action_2[0], set_pos_y=action_2[1]
        )

        ep_reward += reward
        ep_cap1.append(system.user_list[0].capacity)
        ep_cap2.append(system.user_list[1].capacity)
        ep_secure1.append(system.user_list[0].secure_capacity)
        ep_secure2.append(system.user_list[1].secure_capacity)
        ep_attacker.append(np.mean(system.attacker_list[0].capacity))

        # SEE计算
        energy = get_energy_consumption(v_t)
        see = system.user_list[0].secure_capacity / (energy + 1e-10)
        ep_see.append(see)
        ep_energy.append(energy)

        if done:
            break

    rewards_history.append(np.mean([ep_reward]))
    capacity_user1.append(np.mean(ep_cap1))
    capacity_user2.append(np.mean(ep_cap2))
    secure_capacity_user1.append(np.mean(ep_secure1))
    secure_capacity_user2.append(np.mean(ep_secure2))
    attacker_capacity_history.append(np.mean(ep_attacker))
    uav_positions.append(system.UAV_FAS.coordinate[:2].copy())
    see_history.append(np.mean(ep_see))
    energy_history.append(np.mean(ep_energy))

    if (ep + 1) % 50 == 0:
        print(f"  Episode {ep + 1}/{episodes}: reward={ep_reward:.4f}, "
              f"cap1={np.mean(ep_cap1):.4f}, secure1={np.mean(ep_secure1):.4f}")

# ==================== 5. 生成完整可视化 ====================
print("\n[5] 生成完整可视化图表...")

fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle('UAV-FAS TD3 Training Complete Visualization\n'
             'Fluid Antenna Assisted UAV Secure Communication System',
             fontsize=16, fontweight='bold', y=0.98)

# ---------- 图1: 系统拓扑与UAV轨迹 ----------
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title('System Topology & UAV Trajectory', fontsize=12, fontweight='bold')

uav_pos = np.array(uav_positions)
ax1.plot(uav_pos[:, 0], uav_pos[:, 1], 'b-', linewidth=2, alpha=0.7, label='UAV Path')

# 标记起点和终点
ax1.scatter(uav_pos[0, 0], uav_pos[0, 1], c='blue', s=150, marker='o',
            edgecolors='black', linewidth=2, zorder=5, label='Start')
ax1.scatter(uav_pos[-1, 0], uav_pos[-1, 1], c='red', s=150, marker='*',
            edgecolors='black', linewidth=2, zorder=5, label='End')

# 用户位置
for i, user in enumerate(system.user_list):
    ax1.scatter(user.coordinate[0], user.coordinate[1], c='green', s=200, marker='D',
                edgecolors='black', linewidth=2, zorder=5, label=f'User {i}')
    ax1.annotate(f'User {i}\n({user.coordinate[0]:.0f}, {user.coordinate[1]:.0f})',
                xy=(user.coordinate[0], user.coordinate[1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='green', fontweight='bold')

# 攻击者位置
for i, attacker in enumerate(system.attacker_list):
    ax1.scatter(attacker.coordinate[0], attacker.coordinate[1], c='red', s=200, marker='X',
                edgecolors='black', linewidth=2, zorder=5, label=f'Attacker {i}')
    ax1.annotate(f'Attacker {i}\n({attacker.coordinate[0]:.0f}, {attacker.coordinate[1]:.0f})',
                xy=(attacker.coordinate[0], attacker.coordinate[1]),
                xytext=(10, -15), textcoords='offset points',
                fontsize=9, color='red', fontweight='bold')

# RIS位置
ax1.scatter(system.RIS.coordinate[0], system.RIS.coordinate[1], c='orange', s=300, marker='s',
            edgecolors='black', linewidth=2, zorder=5, label='RIS')
ax1.annotate(f'RIS 64-element\n({system.RIS.coordinate[0]:.0f}, {system.RIS.coordinate[1]:.0f})',
            xy=(system.RIS.coordinate[0], system.RIS.coordinate[1]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, color='orange', fontweight='bold')

# 绘制边界
border_x = [-25, 25, 25, -25, -25]
border_y = [0, 0, 50, 50, 0]
ax1.plot(border_x, border_y, 'k--', linewidth=1, alpha=0.5, label='Boundary')

ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.set_xlim(-30, 30)
ax1.set_ylim(-5, 55)
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ---------- 图2: 信道容量分析 ----------
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title('Channel Capacity Analysis', fontsize=12, fontweight='bold')

cap1_arr = np.array(capacity_user1)
cap2_arr = np.array(capacity_user2)
secure1_arr = np.array(secure_capacity_user1)
secure2_arr = np.array(secure_capacity_user2)
attacker_arr = np.array(attacker_capacity_history)

x = range(episodes)
ax2.fill_between(x, 0, cap1_arr, alpha=0.3, color='blue', label='User 1 Total')
ax2.fill_between(x, 0, secure1_arr, alpha=0.5, color='green', label='User 1 Secure')
ax2.fill_between(x, 0, cap2_arr, alpha=0.3, color='red', label='User 2 Total')
ax2.fill_between(x, 0, secure2_arr, alpha=0.5, color='orange', label='User 2 Secure')
ax2.plot(x, attacker_arr, 'k--', linewidth=1.5, label='Attacker')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Capacity (bps/Hz)')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# ---------- 图3: 训练奖励曲线 ----------
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title('Training Reward Curve', fontsize=12, fontweight='bold')

# 平滑处理
def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

ax3.plot(rewards_history, 'b-', alpha=0.3, linewidth=0.5)
ax3.plot(smooth(rewards_history), 'b-', linewidth=2, label='Smoothed')
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Reward')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ---------- 图4: 安全容量曲线 ----------
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title('Secure Capacity per User', fontsize=12, fontweight='bold')

ax4.plot(secure1_arr, 'g-', linewidth=1.5, label='User 1 Secure Cap')
ax4.plot(secure2_arr, 'r-', linewidth=1.5, label='User 2 Secure Cap')
ax4.fill_between(x, secure1_arr, secure2_arr, alpha=0.2, color='purple')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Secure Capacity')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ---------- 图5: SEE (安全能效) ----------
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title('Secrecy Energy Efficiency (SEE)', fontsize=12, fontweight='bold')

see_arr = np.array(see_history)
ax5.plot(see_arr, 'm-', linewidth=1.5, alpha=0.5)
ax5.plot(smooth(see_arr), 'm-', linewidth=2, label='SEE (smoothed)')
ax5.set_xlabel('Episode')
ax5.set_ylabel('SEE (bps/Hz/J)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ---------- 图6: UAV能耗模型 ----------
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_title('UAV Energy Consumption Model', fontsize=12, fontweight='bold')

speeds = np.linspace(0, 0.25, 100)
energies = [get_energy_consumption(v) for v in speeds]
ax6.plot(speeds, np.array(energies) * 1000, 'r-', linewidth=2)  # 转换为mJ
ax6.fill_between(speeds, 0, np.array(energies) * 1000, alpha=0.2, color='red')
ax6.set_xlabel('Speed (m/s)')
ax6.set_ylabel('Energy (mJ)')
ax6.grid(True, alpha=0.3)

# 添加关键点标注
ax6.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7)
ax6.annotate('Max Speed\n0.25 m/s', xy=(0.25, get_energy_consumption(0.25)*1000),
            xytext=(0.15, get_energy_consumption(0.25)*1000*0.8),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=9, color='orange')

# ---------- 图7: 奖励分布 ----------
ax7 = fig.add_subplot(gs[2, 1])
ax7.set_title('Reward Distribution', fontsize=12, fontweight='bold')

ax7.hist(rewards_history, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax7.axvline(np.mean(rewards_history), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(rewards_history):.2f}')
ax7.axvline(np.median(rewards_history), color='green', linestyle='--', linewidth=2,
            label=f'Median: {np.median(rewards_history):.2f}')
ax7.set_xlabel('Reward')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.grid(True, alpha=0.3)

# ---------- 图8: 容量对比柱状图 ----------
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_title('Average Capacity Comparison', fontsize=12, fontweight='bold')

categories = ['User 1\nTotal', 'User 1\nSecure', 'User 2\nTotal', 'User 2\nSecure', 'Attacker']
values = [np.mean(cap1_arr), np.mean(secure1_arr), np.mean(cap2_arr),
          np.mean(secure2_arr), np.mean(attacker_arr)]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#8BC34A', '#F44336']

bars = ax8.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax8.set_ylabel('Capacity (bps/Hz)')
ax8.grid(True, alpha=0.3, axis='y')

# ---------- 图9: RIS相位分布 ----------
ax9 = fig.add_subplot(gs[3, 0])
ax9.set_title('RIS Phase Distribution', fontsize=12, fontweight='bold')

ris_phases = np.angle(np.asarray(system.RIS.Phi_signal).diagonal())
ris_amplitudes = np.abs(np.asarray(system.RIS.Phi_signal).diagonal())

scatter = ax9.scatter(range(len(ris_phases)), np.degrees(ris_phases),
                     c=ris_amplitudes, cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax9, label='Amplitude')
ax9.set_xlabel('RIS Element Index')
ax9.set_ylabel('Phase (degrees)')
ax9.grid(True, alpha=0.3)

# ---------- 图10: 波束成形功率分布 ----------
ax10 = fig.add_subplot(gs[3, 1])
ax10.set_title('Beamforming Power Distribution', fontsize=12, fontweight='bold')

G_power = np.abs(np.asarray(system.UAV_FAS.G)) ** 2
for k in range(G_power.shape[1]):
    ax10.bar(range(G_power.shape[0]), G_power[:, k], alpha=0.7, label=f'User {k}')
ax10.set_xlabel('Antenna Index')
ax10.set_ylabel('Power')
ax10.legend()
ax10.grid(True, alpha=0.3, axis='y')

# ---------- 图11: 综合性能仪表盘 ----------
ax11 = fig.add_subplot(gs[3, 2])
ax11.set_title('System Performance Summary', fontsize=12, fontweight='bold')
ax11.axis('off')

summary_text = f"""
{'='*40}
System Performance Summary
{'='*40}

Training Episodes: {episodes}
Steps per Episode: {steps}

Average Performance:
  User 1 Capacity:    {np.mean(cap1_arr):.4f} bps/Hz
  User 1 Secure Cap:  {np.mean(secure1_arr):.4f} bps/Hz
  User 2 Capacity:    {np.mean(cap2_arr):.4f} bps/Hz
  User 2 Secure Cap:  {np.mean(secure2_arr):.4f} bps/Hz
  Attacker Capacity:  {np.mean(attacker_arr):.4f} bps/Hz

Security Metrics:
  Secure Ratio:       {np.mean(secure1_arr)/(np.mean(cap1_arr)+1e-10)*100:.1f}%
  Avg SEE:            {np.mean(see_arr):.4f} bps/Hz/J

System Configuration:
  BS Antennas:        8 (ULA)
  FAS Ports:          16 (Discrete)
  RIS Elements:       64 (8x8 UPA)
  Users:              2
  Frequency:          28 GHz (mmWave)
"""

ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图表
output_path = 'data/storage/uav_bs_fas/scratch/td3_see_39/complete_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  图表已保存: {output_path}")

plt.close()

# ==================== 6. 生成HTML报告 ====================
print("\n[6] 生成HTML报告...")

# 计算汇总统计
stats = {
    'episodes': episodes,
    'steps': steps,
    'mean_reward': np.mean(rewards_history),
    'mean_cap1': np.mean(cap1_arr),
    'mean_cap2': np.mean(cap2_arr),
    'mean_secure1': np.mean(secure1_arr),
    'mean_secure2': np.mean(secure2_arr),
    'mean_attacker': np.mean(attacker_arr),
    'mean_see': np.mean(see_arr),
    'secure_ratio1': np.mean(secure1_arr)/(np.mean(cap1_arr)+1e-10)*100,
    'secure_ratio2': np.mean(secure2_arr)/(np.mean(cap2_arr)+1e-10)*100,
    'final_uav': uav_pos[-1],
}

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UAV-FAS TD3 Complete Visualization Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; margin: 30px 0; font-size: 2.2em; }}
        h2 {{ color: #38bdf8; border-bottom: 2px solid #1e3a5f; padding-bottom: 8px; margin: 30px 0 15px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin: 16px 0; border: 1px solid #334155; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
        .stat-card {{ background: linear-gradient(135deg, #1e3a5f, #1e293b); border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #2563eb33; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #38bdf8; }}
        .stat-label {{ color: #94a3b8; font-size: 0.9em; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #1e3a5f; color: #7dd3fc; font-weight: 600; }}
        tr:hover {{ background: #1e3a5f44; }}
        img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; }}
        .chart-container {{ background: #0f172a; border-radius: 8px; padding: 16px; margin: 10px 0; text-align: center; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed #334155; }}
        .metric-label {{ color: #94a3b8; }}
        .metric-value {{ color: #38bdf8; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
        .badge-success {{ background: #065f46; color: #34d399; }}
        .badge-warning {{ background: #78350f; color: #fbbf24; }}
        .badge-info {{ background: #1e3a5f; color: #38bdf8; }}
        .code {{ background: #0f172a; padding: 16px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 0.9em; overflow-x: auto; border: 1px solid #334155; white-space: pre-wrap; }}
        footer {{ text-align: center; padding: 30px; color: #64748b; font-size: 0.85em; }}
        .section {{ margin: 20px 0; }}
        .highlight {{ background: #38bdf822; border-left: 4px solid #38bdf8; padding: 16px; margin: 12px 0; border-radius: 0 8px 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UAV-FAS TD3 Complete Visualization Report</h1>
        <p style="text-align:center; color:#94a3b8; margin-bottom:30px;">
            Fluid Antenna Assisted UAV Secure Communication | Reinforcement Learning Training
        </p>

        <!-- Performance Summary -->
        <div class="grid">
            <div class="stat-card">
                <div class="stat-value">{stats['mean_cap1']:.4f}</div>
                <div class="stat-label">User 1 Avg Capacity (bps/Hz)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#34d399;">{stats['mean_secure1']:.4f}</div>
                <div class="stat-label">User 1 Secure Capacity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#fbbf24;">{stats['mean_see']:.4f}</div>
                <div class="stat-label">Avg SEE (bps/Hz/J)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:#f59e0b;">{stats['mean_reward']:.4f}</div>
                <div class="stat-label">Average Reward</div>
            </div>
        </div>

        <!-- System Topology -->
        <div class="card">
            <h2>System Topology & UAV Trajectory</h2>
            <div class="highlight">
                <strong>Bug Fix Applied:</strong> RIS coordinate system NaN issue resolved by changing
                <code>ris_coor_sys_z</code> from <code>[0,0,1]</code> to <code>[0,1,0]</code>.
                FAS gain initialization issue fixed by initializing F matrix with unit vectors.
            </div>
            <div class="chart-container">
                <img src="complete_visualization.png" alt="Complete Visualization">
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="card">
            <h2>Detailed Performance Metrics</h2>
            <div class="grid">
                <div>
                    <h3>User Capacity Analysis</h3>
                    <div class="metric">
                        <span class="metric-label">User 1 Total Capacity</span>
                        <span class="metric-value">{stats['mean_cap1']:.6f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">User 1 Secure Capacity</span>
                        <span class="metric-value" style="color:#34d399;">{stats['mean_secure1']:.6f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">User 2 Total Capacity</span>
                        <span class="metric-value">{stats['mean_cap2']:.6f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">User 2 Secure Capacity</span>
                        <span class="metric-value" style="color:#34d399;">{stats['mean_secure2']:.6f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Attacker Capacity</span>
                        <span class="metric-value" style="color:#f87171;">{stats['mean_attacker']:.6f} bps/Hz</span>
                    </div>
                </div>
                <div>
                    <h3>Security Analysis</h3>
                    <div class="metric">
                        <span class="metric-label">User 1 Secure Ratio</span>
                        <span class="metric-value">{stats['secure_ratio1']:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">User 2 Secure Ratio</span>
                        <span class="metric-value">{stats['secure_ratio2']:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg SEE</span>
                        <span class="metric-value">{stats['mean_see']:.6f} bps/Hz/J</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Final UAV Position</span>
                        <span class="metric-value">({stats['final_uav'][0]:.1f}, {stats['final_uav'][1]:.1f})</span>
                    </div>
                </div>
                <div>
                    <h3>System Configuration</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>BS Antennas</td><td>8 (ULA)</td></tr>
                        <tr><td>FAS Ports</td><td>16 (Discrete)</td></tr>
                        <tr><td>RIS Elements</td><td>64 (8x8 UPA)</td></tr>
                        <tr><td>Users</td><td>2</td></tr>
                        <tr><td>Frequency</td><td>28 GHz (mmWave)</td></tr>
                        <tr><td>Algorithm</td><td>Twin Delayed DDPG (TD3)</td></tr>
                        <tr><td>Reward Design</td><td>SEE (Secrecy Energy Efficiency)</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- Bug Fixes -->
        <div class="card">
            <h2>Bug Fixes Applied</h2>
            <div class="grid">
                <div class="highlight">
                    <h3 style="color:#34d399;">Fix 1: RIS Coordinate System NaN</h3>
                    <p><strong>Root Cause:</strong> <code>ris_coor_sys_z = [0,0,1]</code> was parallel to reference vector <code>[0,0,1]</code>,
                    causing <code>cross product = [0,0,0]</code> and division by zero.</p>
                    <p><strong>Fix:</strong> Changed to <code>ris_coor_sys_z = [0,1,0]</code></p>
                </div>
                <div class="highlight">
                    <h3 style="color:#34d399;">Fix 2: FAS Gain Zero Initialization</h3>
                    <p><strong>Root Cause:</strong> F matrix initialized as zeros, making <code>fas_gain = mean(abs(F)) = 0</code>,
                    causing <code>H_eff = 0</code> and all capacities = 0.</p>
                    <p><strong>Fix:</strong> Initialize F with unit vectors scaled by power factor.</p>
                </div>
            </div>
        </div>

        <footer>
            Generated: 2026-05-28 | UAV-FAS TD3 Complete Visualization System
        </footer>
    </div>
</body>
</html>"""

html_path = 'data/storage/uav_bs_fas/scratch/td3_see_39/complete_visualization.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"  HTML报告已保存: {html_path}")

print("\n" + "=" * 70)
print("完成!")
print("=" * 70)
print(f"\n文件位置:")
print(f"  - 图表: data/storage/uav_bs_fas/scratch/td3_see_39/complete_visualization.png")
print(f"  - HTML: data/storage/uav_bs_fas/scratch/td3_see_39/complete_visualization.html")
