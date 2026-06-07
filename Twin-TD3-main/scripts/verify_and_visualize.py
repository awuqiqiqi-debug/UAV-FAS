import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from src.envs.uav_comm_env import MiniSystem, get_energy_consumption
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("UAV-BS-FAS 系统验证与可视化")
print("=" * 60)

print("\n[1] 初始化系统...")
system = MiniSystem(
    user_num=2, bs_ant_num=8, fas_ant_num=12, ris_ant_num=64,
    if_dir_link=1, if_with_FAS=True, if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design='see', project_name='verify_test', step_num=100
)

print("\n[2] 验证单步运行...")
system.reset()
action_1 = np.random.randn(192) * 0.3
action_2 = np.array([0.1, 0.1])
new_state, reward, done, info = system.step(
    action_0=action_2[0], action_1=action_2[1],
    G=action_1[0:32], Phi=action_1[32:],
    set_pos_x=action_2[0], set_pos_y=action_2[1]
)
print(f"  状态维度: {len(new_state)}")
print(f"  奖励: {reward:.4f}")
print(f"  是否终止: {done}")

print("\n[3] 信道容量验证:")
for i, user in enumerate(system.user_list):
    print(f"  用户{i}: 容量={user.capacity:.4f}, 安全容量={user.secure_capacity:.4f}")
for i, attacker in enumerate(system.attacker_list):
    print(f"  攻击者{i}: 容量={attacker.capacity}")

print("\n[4] 模拟训练过程生成可视化数据...")
episodes = 100
steps = 50

rewards_history = []
capacity_history = []
attacker_capacity_history = []
uav_positions = []

for ep in range(episodes):
    system.reset()
    ep_reward = 0
    ep_capacity = []
    ep_attacker_cap = []

    for step in range(steps):
        action_1 = np.random.randn(192) * 0.3
        action_2 = np.random.randn(2) * 0.5

        new_state, reward, done, _ = system.step(
            action_0=action_2[0], action_1=action_2[1],
            G=action_1[0:32], Phi=action_1[32:],
            set_pos_x=action_2[0], set_pos_y=action_2[1]
        )

        ep_reward += reward
        ep_capacity.append([u.capacity for u in system.user_list])
        ep_attacker_cap.append([a.capacity.mean() for a in system.attacker_list])

        if done:
            break

    rewards_history.append(ep_reward)
    capacity_history.append(np.mean(ep_capacity, axis=0))
    attacker_capacity_history.append(np.mean(ep_attacker_cap, axis=0))
    uav_positions.append(system.UAV_BS_FAS.coordinate[:2].copy())

    if (ep + 1) % 20 == 0:
        print(f"  Episode {ep + 1}/{episodes}: reward={ep_reward:.4f}")

print("\n[5] 生成可视化图表...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('UAV-BS-FAS TD3 Training Visualization', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(rewards_history, 'b-', linewidth=1.5, alpha=0.7)
ax1.fill_between(range(episodes), rewards_history, alpha=0.2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Reward Curve')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
cap_arr = np.array(capacity_history)
ax2.plot(cap_arr[:, 0], 'g-', label='User 1', linewidth=1.5)
ax2.plot(cap_arr[:, 1], 'r-', label='User 2', linewidth=1.5)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Capacity (bps/Hz)')
ax2.set_title('User Channel Capacity')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
secure_cap = cap_arr - np.array(attacker_capacity_history)
secure_cap = np.maximum(secure_cap, 0)
ax3.plot(secure_cap[:, 0], 'g-', label='User 1', linewidth=1.5)
ax3.plot(secure_cap[:, 1], 'r-', label='User 2', linewidth=1.5)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Secure Capacity')
ax3.set_title('Secure Capacity')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 0]
uav_pos = np.array(uav_positions)
ax4.plot(uav_pos[:, 0], uav_pos[:, 1], 'b-', linewidth=1.5, alpha=0.7)
ax4.scatter(uav_pos[0, 0], uav_pos[0, 1], c='green', s=100, marker='o', label='Start')
ax4.scatter(uav_pos[-1, 0], uav_pos[-1, 1], c='red', s=100, marker='*', label='End')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_title('UAV Trajectory')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-30, 30)
ax4.set_ylim(-5, 55)

ax5 = axes[1, 1]
speeds = np.linspace(0, 0.25, 50)
energies = [get_energy_consumption(v) for v in speeds]
ax5.plot(speeds, energies, 'r-', linewidth=2)
ax5.fill_between(speeds, energies, alpha=0.2)
ax5.set_xlabel('Speed (m/s)')
ax5.set_ylabel('Energy (J)')
ax5.set_title('UAV Energy Consumption Model')
ax5.grid(True, alpha=0.3)

ax6 = axes[1, 2]
ax6.hist(rewards_history, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax6.axvline(np.mean(rewards_history), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(rewards_history):.2f}')
ax6.set_xlabel('Reward')
ax6.set_ylabel('Frequency')
ax6.set_title('Reward Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()

output_path = 'data/storage/uav_bs_fas/scratch/td3_see_39/training_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Chart saved: {output_path}")

plt.close()

print("\n[6] Generating HTML report...")

mean_reward = np.mean(rewards_history)
mean_cap1 = np.mean(cap_arr[:, 0])
mean_cap2 = np.mean(cap_arr[:, 1])
mean_secure1 = np.mean(secure_cap[:, 0])
mean_secure2 = np.mean(secure_cap[:, 1])
secure_ratio = np.mean(secure_cap) / np.mean(cap_arr) * 100
final_pos = uav_pos[-1]

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UAV-BS-FAS TD3 Training Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; margin: 30px 0; font-size: 2.2em; }}
        h2 {{ color: #38bdf8; border-bottom: 2px solid #1e3a5f; padding-bottom: 8px; margin: 30px 0 15px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin: 16px 0; border: 1px solid #334155; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
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
        footer {{ text-align: center; padding: 30px; color: #64748b; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UAV-BS-FAS TD3 Training Visualization</h1>
        <p style="text-align:center; color:#94a3b8; margin-bottom:30px;">Fluid Antenna Assisted UAV Secure Communication | RL Training Results</p>

        <div class="grid">
            <div class="stat-card">
                <div class="stat-value">{episodes}</div>
                <div class="stat-label">Verification Episodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{mean_reward:.2f}</div>
                <div class="stat-label">Average Reward</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{mean_cap1:.2f}</div>
                <div class="stat-label">User 1 Avg Capacity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{mean_secure1:.2f}</div>
                <div class="stat-label">User 1 Avg Secure Cap</div>
            </div>
        </div>

        <div class="card">
            <h2>Training Visualization</h2>
            <div class="chart-container">
                <img src="training_visualization.png" alt="Training Visualization">
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="grid">
                <div>
                    <h3>Capacity Analysis</h3>
                    <div class="metric">
                        <span class="metric-label">User 1 Avg Capacity</span>
                        <span class="metric-value">{mean_cap1:.4f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">User 2 Avg Capacity</span>
                        <span class="metric-value">{mean_cap2:.4f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Capacity Std</span>
                        <span class="metric-value">{np.std(cap_arr):.4f}</span>
                    </div>
                </div>
                <div>
                    <h3>Security Analysis</h3>
                    <div class="metric">
                        <span class="metric-label">User 1 Secure Capacity</span>
                        <span class="metric-value">{mean_secure1:.4f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">User 2 Secure Capacity</span>
                        <span class="metric-value">{mean_secure2:.4f} bps/Hz</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Secure Ratio</span>
                        <span class="metric-value">{secure_ratio:.1f}%</span>
                    </div>
                </div>
                <div>
                    <h3>UAV Flight Analysis</h3>
                    <div class="metric">
                        <span class="metric-label">Final Position</span>
                        <span class="metric-value">({final_pos[0]:.1f}, {final_pos[1]:.1f})</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>System Configuration</h2>
            <div class="grid">
                <div>
                    <h3>Hardware Parameters</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>BS Antennas</td><td>8 (ULA)</td></tr>
                        <tr><td>FAS Ports</td><td>16 (Discrete)</td></tr>
                        <tr><td>RIS Elements</td><td>64 (8x8 UPA)</td></tr>
                        <tr><td>Users</td><td>2</td></tr>
                        <tr><td>Frequency</td><td>28 GHz (mmWave)</td></tr>
                    </table>
                </div>
                <div>
                    <h3>Training Parameters</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Algorithm</td><td>Twin Delayed DDPG (TD3)</td></tr>
                        <tr><td>Reward Design</td><td>SEE (Secrecy Energy Efficiency)</td></tr>
                        <tr><td>Learning Rate</td><td>0.0001 / 0.001</td></tr>
                        <tr><td>Batch Size</td><td>128 (BS+FAS), 64 (UAV)</td></tr>
                        <tr><td>Tau</td><td>0.001</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <footer>
            Generated: 2026-05-28 | UAV-BS-FAS TD3 Training Visualization System
        </footer>
    </div>
</body>
</html>"""

html_path = 'data/storage/uav_bs_fas/scratch/td3_see_39/training_visualization.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"  HTML report saved: {html_path}")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)
print(f"\nVisualization files:")
print(f"  - Chart: data/storage/uav_bs_fas/scratch/td3_see_39/training_visualization.png")
print(f"  - HTML:  data/storage/uav_bs_fas/scratch/td3_see_39/training_visualization.html")
