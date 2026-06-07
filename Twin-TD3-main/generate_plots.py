"""Generate all plots for the v15-style HTML viewer"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.io, os, numpy as np

# Try to use Chinese font
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

path = 'data/storage/scratch/td3_mfris_fas_v1_2'
plot_dir = os.path.join(path, 'plot')
ris_pos = np.array([12.0, 35.0])
uav_start = np.array([-10.0, 10.0])

# Collect all episode data
all_rewards = []
all_sec = []
all_att = []
all_dists = []
all_user_caps = []
all_ris_skd = []
all_fas_ports = []
ep_traj = {}

for ep in range(501):
    f = os.path.join(path, f'simulation_result_ep_{ep}.mat')
    if not os.path.exists(f): continue
    data = scipy.io.loadmat(f)
    result = data[f'result_{ep}']

    uav_state = result['UAV_state'][0,0]
    reward = result['reward'][0,0]
    sec_cap = result['secure_capacity'][0,0]
    att_cap = result['attaker_capacity'][0,0]
    user_cap = result['user_capacity'][0,0]
    ris_skd = result['RIS_scheduling'][0,0]
    fas_port = result['FAS_port'][0,0]

    end_pos = uav_state[-1, :2]
    end_dist = float(np.linalg.norm(end_pos - ris_pos))

    all_rewards.append(float(np.mean(reward)))
    all_sec.append(float(np.mean(sec_cap)))
    all_att.append(float(np.mean(att_cap)))
    all_user_caps.append(float(np.mean(user_cap)))
    all_dists.append(end_dist)
    all_ris_skd.append(float(np.mean(ris_skd)))
    from scipy.stats import mode as sp_mode
    all_fas_ports.append(int(sp_mode(fas_port.flatten(), keepdims=False).mode))

    if ep in [0, 1, 50, 100, 200, 300, 400, 500]:
        ep_traj[ep] = uav_state[:, :2].copy()

print(f"Collected {len(all_rewards)} episodes")

def smooth(data, w=20):
    return np.convolve(data, np.ones(w)/w, mode='valid')

eps = np.arange(len(all_rewards))

# ========== 1. Trajectory plot ==========
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')

# Plot multiple trajectories
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ep_traj)))
for (ep, traj), color in zip(sorted(ep_traj.items()), colors):
    alpha = 0.4 + 0.6 * (ep / 500)
    lw = 0.8 + 1.2 * (ep / 500)
    ax.plot(traj[:,0], traj[:,1], '-', color=color, alpha=alpha, linewidth=lw, label=f'ep {ep}')
    ax.plot(traj[-1,0], traj[-1,1], 'o', color=color, markersize=6, alpha=0.8)

# RIS
ax.plot(ris_pos[0], ris_pos[1], '^', color='#ffaa00', markersize=18, zorder=10, label='RIS (12,35)')
ax.annotate('RIS', (ris_pos[0], ris_pos[1]), textcoords='offset points', xytext=(8, 8),
            fontsize=14, color='#ffaa00', fontweight='bold')

# Users
users = [(4, 47), (25, 25)]
for i, (ux, uy) in enumerate(users):
    ax.plot(ux, uy, 's', color='#4ade80', markersize=12, zorder=10)
    ax.annotate(f'User {i}', (ux, uy), textcoords='offset points', xytext=(8, 5),
                fontsize=10, color='#4ade80')

# Eavesdropper
ax.plot(20, 10, 'X', color='#f87171', markersize=14, zorder=10, label='Eavesdropper')
ax.annotate('Eve', (20, 10), textcoords='offset points', xytext=(8, 5),
            fontsize=10, color='#f87171')

# UAV start
ax.plot(uav_start[0], uav_start[1], 'D', color='white', markersize=12, zorder=10)
ax.annotate('UAV Start', (uav_start[0], uav_start[1]), textcoords='offset points', xytext=(8, -12),
            fontsize=10, color='white')

ax.set_xlim(-30, 30)
ax.set_ylim(-5, 55)
ax.set_xlabel('X (m)', color='#94a3b8', fontsize=12)
ax.set_ylabel('Y (m)', color='#94a3b8', fontsize=12)
ax.set_title('UAV Trajectory Evolution (Multi-functional RIS + FAS)', color='#60a5fa', fontsize=14, pad=15)
ax.legend(loc='upper left', fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'trajectory.png'), dpi=150, facecolor='#0f172a', bbox_inches='tight')
plt.close()
print("Saved trajectory.png")

# ========== 2. Reward curve ==========
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')
s = smooth(all_rewards)
ax.plot(eps[:len(s)], s, color='#60a5fa', linewidth=2)
ax.fill_between(eps[:len(s)], s, alpha=0.1, color='#60a5fa')
ax.set_xlabel('Episode', color='#94a3b8')
ax.set_ylabel('Reward', color='#94a3b8')
ax.set_title('Average Reward per Episode (Smoothed)', color='#60a5fa', fontsize=13, pad=10)
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'reward.png'), dpi=150, facecolor='#0f172a')
plt.close()
print("Saved reward.png")

# ========== 3. Secure capacity ==========
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')
s = smooth(all_sec)
ax.plot(eps[:len(s)], s, color='#4ade80', linewidth=2)
ax.fill_between(eps[:len(s)], s, alpha=0.1, color='#4ade80')
ax.set_xlabel('Episode', color='#94a3b8')
ax.set_ylabel('Secure Capacity (bits/s/Hz)', color='#94a3b8')
ax.set_title('Average Secure Capacity per Episode (Smoothed)', color='#4ade80', fontsize=13, pad=10)
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'secure_capacity.png'), dpi=150, facecolor='#0f172a')
plt.close()
print("Saved secure_capacity.png")

# ========== 4. User capacity ==========
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')
s = smooth(all_user_caps)
ax.plot(eps[:len(s)], s, color='#60a5fa', linewidth=2)
ax.fill_between(eps[:len(s)], s, alpha=0.1, color='#60a5fa')
ax.set_xlabel('Episode', color='#94a3b8')
ax.set_ylabel('User Capacity (bits/s/Hz)', color='#94a3b8')
ax.set_title('Average User Channel Capacity (Smoothed)', color='#60a5fa', fontsize=13, pad=10)
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'user_capacity.png'), dpi=150, facecolor='#0f172a')
plt.close()
print("Saved user_capacity.png")

# ========== 5. Attacker capacity ==========
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')
s = smooth(all_att)
ax.plot(eps[:len(s)], s, color='#f87171', linewidth=2)
ax.fill_between(eps[:len(s)], s, alpha=0.1, color='#f87171')
ax.set_xlabel('Episode', color='#94a3b8')
ax.set_ylabel('Attacker Capacity (bits/s/Hz)', color='#94a3b8')
ax.set_title('Average Eavesdropper Capacity (Smoothed) - Lower is Better', color='#f87171', fontsize=13, pad=10)
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'attaker_capacity.png'), dpi=150, facecolor='#0f172a')
plt.close()
print("Saved attaker_capacity.png")

# ========== 6. RIS distance ==========
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')
s = smooth(all_dists)
ax.plot(eps[:len(s)], s, color='#fb923c', linewidth=2)
ax.fill_between(eps[:len(s)], s, alpha=0.1, color='#fb923c')
ax.axhline(y=0, color='#334155', linestyle='--', alpha=0.5)
ax.set_xlabel('Episode', color='#94a3b8')
ax.set_ylabel('Distance to RIS (m)', color='#94a3b8')
ax.set_title('UAV-RIS Distance at End of Episode (Smoothed)', color='#fb923c', fontsize=13, pad=10)
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'ris_distance.png'), dpi=150, facecolor='#0f172a')
plt.close()
print("Saved ris_distance.png")

# ========== 7. RIS scheduling ==========
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#0f172a')
fig.patch.set_facecolor('#0f172a')
s = smooth(all_ris_skd)
ax.plot(eps[:len(s)], s, color='#c084fc', linewidth=2)
ax.fill_between(eps[:len(s)], s, alpha=0.1, color='#c084fc')
ax.set_xlabel('Episode', color='#94a3b8')
ax.set_ylabel('Scheduling Rate', color='#94a3b8')
ax.set_title('RIS Element Scheduling Rate (Reflect vs Jam)', color='#c084fc', fontsize=13, pad=10)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.15, color='#334155')
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values(): spine.set_color('#334155')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'ris_scheduling.png'), dpi=150, facecolor='#0f172a')
plt.close()
print("Saved ris_scheduling.png")

# ========== 8. RIS phase plots (from last episode) ==========
f_last = os.path.join(path, 'simulation_result_ep_500.mat')
if os.path.exists(f_last):
    data = scipy.io.loadmat(f_last)
    result = data['result_500']
    reflect_coeff = result['reflecting_coefficient'][0,0]  # (T, 4)

    for elem in range(4):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        phase = np.angle(reflect_coeff[:, elem])
        magnitude = np.abs(reflect_coeff[:, elem])
        ax.plot(phase, color='#c084fc', linewidth=1.5, label='Phase')
        ax2 = ax.twinx()
        ax2.plot(magnitude, color='#60a5fa', linewidth=1, alpha=0.5, label='Magnitude')
        ax.set_xlabel('Step', color='#94a3b8')
        ax.set_ylabel('Phase (rad)', color='#c084fc')
        ax2.set_ylabel('Magnitude', color='#60a5fa')
        ax.set_title(f'RIS Element #{elem} Phase & Magnitude (ep 500)', color='#93c5fd', fontsize=12, pad=10)
        ax.grid(True, alpha=0.15, color='#334155')
        ax.tick_params(colors='#94a3b8')
        ax2.tick_params(colors='#94a3b8')
        for spine in ax.spines.values(): spine.set_color('#334155')
        for spine in ax2.spines.values(): spine.set_color('#334155')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'RIS', f'RIS_{elem}_element.png'), dpi=150, facecolor='#0f172a')
        plt.close()
        print(f"Saved RIS_{elem}_element.png")

print("\nAll plots generated!")
