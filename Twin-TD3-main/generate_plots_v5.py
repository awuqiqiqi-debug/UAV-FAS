"""Generate all plots for td3_mfris_jam_v5 — matching reference style exactly"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io, os, numpy as np, math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

path = 'data/storage/scratch/td3_mfris_jam_v5'
plot_dir = os.path.join(path, 'plot')
os.makedirs(os.path.join(plot_dir, 'RIS'), exist_ok=True)
ris_pos = np.array([12.0, 35.0])
uav_start = np.array([-10.0, 10.0])

# Energy parameters (from env.py)
P_0 = 78.7750
U2_tip = 200
d_0 = 0.6
p = 1.225
s = 0.05
A = 0.503
P_i = 88.2250
delta_time = 1

def get_energy(v_t):
    e1 = P_0 + 3*P_0*(abs(v_t))**2/U2_tip + 0.5*d_0*p*s*A*(abs(v_t))**3
    v_0 = (P_0*4/(A*2*p))**0.5
    e2 = P_i*((1+(abs(v_t)**4)/(4*v_0**4))**0.5 - (abs(v_t)**2)/(2*v_0**2))**0.5
    return delta_time*(e1+e2)

all_user_caps_per_ep = []
all_sec_per_ep = []
all_att_per_ep = []
all_rewards = []
all_dists = []
all_ris_skd = []
all_ssr = []       # sum secrecy rate
all_see = []       # secrecy energy efficiency
ep_traj = {}
sample_eps = [0, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000]

for ep in range(2001):
    f = os.path.join(path, f'simulation_result_ep_{ep}.mat')
    if not os.path.exists(f): continue
    data = scipy.io.loadmat(f)
    result = data[f'result_{ep}']
    uav_state = result['UAV_state'][0,0]
    all_rewards.append(float(np.mean(result['reward'][0,0])))
    uc = np.array(result['user_capacity'][0,0])        # (100, 2)
    sc = np.array(result['secure_capacity'][0,0])      # (100, 2)
    ac = np.array(result['attaker_capacity'][0,0])
    um = np.array(result['UAV_movement'][0,0])         # (100, 2)
    all_user_caps_per_ep.append(np.mean(uc, axis=0))
    all_sec_per_ep.append(np.mean(sc, axis=0))
    all_att_per_ep.append(float(np.mean(ac)))
    # SSR = sum of secure_capacity per step, averaged over steps
    ssr_per_step = np.sum(sc, axis=1)  # (100,)
    all_ssr.append(float(np.mean(ssr_per_step)))
    # SEE = SSR / total_energy
    total_energy = sum(get_energy(np.linalg.norm(um[t])) for t in range(len(um)))
    see = float(np.mean(ssr_per_step) / (total_energy + 1e-10)) * 1000  # bits/s/Hz/kJ
    all_see.append(see)
    all_dists.append(float(np.linalg.norm(uav_state[-1, :2] - ris_pos)))
    all_ris_skd.append(float(np.mean(result['RIS_scheduling'][0,0])))
    if ep in sample_eps:
        ep_traj[ep] = uav_state[:, :2].copy()

def smooth(data, w=30):
    return np.convolve(data, np.ones(w)/w, mode='valid')

eps = np.arange(len(all_rewards))
user_caps_arr = np.array(all_user_caps_per_ep)
sec_arr = np.array(all_sec_per_ep)
att_arr = np.array(all_att_per_ep)
ssr_arr = np.array(all_ssr)
see_arr = np.array(all_see)

print(f"Total episodes: {len(all_rewards)}")
for k in range(2):
    print(f"User {k}: max={np.max(user_caps_arr[:,k]):.3f}, final={np.mean(user_caps_arr[-50:,k]):.3f}")
print(f"Attacker: min={np.min(att_arr):.3f}, final={np.mean(att_arr[-50:]):.3f}")
print(f"SSR: max={np.max(ssr_arr):.3f}, final={np.mean(ssr_arr[-50:]):.3f}")
print(f"SEE: max={np.max(see_arr):.3f}, final={np.mean(see_arr[-50:]):.3f}")

# ===== Reference-matched style =====
BG = '#ffffff'
GRID_COLOR = '#e8e8e8'
TICK_COLOR = '#444444'
LABEL_SIZE = 18
TICK_SIZE = 14
TITLE_SIZE = 18
LEGEND_SIZE = 11
LINE_W = 2.2
# Reference uses very light fill: same hue as line but very low alpha
FILL_ALPHA = 0.10
C_USER0 = '#1976d2'   # darker blue (reference User 0)
C_USER1 = '#ff6d00'   # orange-red (reference User 1)
C_ATT = '#388e3c'     # green (reference Attacker)

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.35, color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(axis='both', colors=TICK_COLOR, labelsize=TICK_SIZE, length=4, width=0.8)
    for s in ax.spines.values():
        s.set_color('#dddddd'); s.set_linewidth(0.7)

# 1. Trajectory
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('#0f172a'); fig.patch.set_facecolor('#0f172a')
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ep_traj)))
for (ep, traj), color in zip(sorted(ep_traj.items()), colors):
    a = 0.3 + 0.7 * (ep / 2000)
    lw = 0.8 + 1.5 * (ep / 2000)
    ax.plot(traj[:,0], traj[:,1], '-', color=color, alpha=a, linewidth=lw, label=f'ep {ep}')
    ax.plot(traj[-1,0], traj[-1,1], 'o', color=color, markersize=6, alpha=0.8)
ax.plot(ris_pos[0], ris_pos[1], '^', color='#ffaa00', markersize=18, zorder=10)
ax.annotate('RIS', (ris_pos[0], ris_pos[1]), textcoords='offset points', xytext=(8,8), fontsize=14, color='#ffaa00', fontweight='bold')
for i, (ux,uy) in enumerate([(4,47),(25,25)]):
    ax.plot(ux, uy, 's', color='#4ade80', markersize=12, zorder=10)
    ax.annotate(f'User {i}', (ux,uy), textcoords='offset points', xytext=(8,5), fontsize=10, color='#4ade80')
ax.plot(20, 10, 'X', color='#f87171', markersize=14, zorder=10)
ax.annotate('Eve', (20,10), textcoords='offset points', xytext=(8,5), fontsize=10, color='#f87171')
ax.plot(uav_start[0], uav_start[1], 'D', color='white', markersize=12, zorder=10)
ax.annotate('UAV Start', (uav_start[0], uav_start[1]), textcoords='offset points', xytext=(8,-12), fontsize=10, color='white')
ax.set_xlim(-30,30); ax.set_ylim(-5,55)
ax.set_xlabel('X (m)', color='#94a3b8', fontsize=LABEL_SIZE); ax.set_ylabel('Y (m)', color='#94a3b8', fontsize=LABEL_SIZE)
ax.set_title('UAV Trajectory - v5 8-elem RIS + Heuristic-only (2000 eps)', color='#60a5fa', fontsize=TITLE_SIZE, pad=15)
ax.legend(loc='upper left', fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
ax.grid(True, alpha=0.15, color='#334155'); ax.tick_params(colors='#94a3b8', labelsize=TICK_SIZE)
for s in ax.spines.values(): s.set_color('#334155')
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'trajectory.png'), dpi=150, facecolor='#0f172a'); plt.close(); print("Saved trajectory.png")

# 2. Reward — dark blue smoothed curve + light fill
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
s = smooth(all_rewards)
ax.plot(eps[:len(s)], s, color=C_USER0, linewidth=LINE_W, label='30-ep Moving Avg')
ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color=C_USER0)
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Reward', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('Average Reward per Episode', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(bottom=0)
ax.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.9, edgecolor='#cccccc')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'reward.png'), dpi=150, facecolor=BG); plt.close(); print("Saved reward.png")

# 2b. Average Sum Secrecy Rate (SSR)
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
s = smooth(ssr_arr)
ax.plot(eps[:len(s)], s, color=C_USER0, linewidth=LINE_W, label='30-ep Moving Avg')
ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color=C_USER0)
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Average SSR (bits/s/Hz)', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('Average Sum Secrecy Rate (SSR) per Episode', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(0, 6)
ax.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.9, edgecolor='#cccccc')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'average_sum_secrecy_rate.png'), dpi=150, facecolor=BG); plt.close(); print("Saved average_sum_secrecy_rate.png")

# 2c. Average Secrecy Energy Efficiency (SEE)
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
s = smooth(see_arr)
ax.plot(eps[:len(s)], s, color='#2e7d32', linewidth=LINE_W, label='30-ep Moving Avg')
ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color='#2e7d32')
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Average SEE (bits/s/Hz/kJ)', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('Average Secrecy Energy Efficiency (SEE) per Episode', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(0, 0.4)
ax.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.9, edgecolor='#cccccc')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'average_secrecy_energy_efficiency.png'), dpi=150, facecolor=BG); plt.close(); print("Saved average_secrecy_energy_efficiency.png")

# 3. Secure capacity — per-user, matching reference exactly
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for k, (c, label) in enumerate([(C_USER0, 'User 0'), (C_USER1, 'User 1')]):
    s = smooth(sec_arr[:, k])
    ax.plot(eps[:len(s)], s, color=c, linewidth=LINE_W, label=label)
    ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color=c)
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Secure Capacity (bits/s/Hz)', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('Secure Capacity per Episode (Averaged)', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(0, 5)
ax.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.9, edgecolor='#cccccc')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'secure_capacity.png'), dpi=150, facecolor=BG); plt.close(); print("Saved secure_capacity.png")

# 4. User capacity — per-user
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for k, (c, label) in enumerate([(C_USER0, 'User 0'), (C_USER1, 'User 1')]):
    s = smooth(user_caps_arr[:, k])
    ax.plot(eps[:len(s)], s, color=c, linewidth=LINE_W, label=label)
    ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color=c)
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('User Capacity (bits/s/Hz)', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('User Channel Capacity per Episode (Averaged)', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(0, 5)
ax.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.9, edgecolor='#cccccc')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'user_capacity.png'), dpi=150, facecolor=BG); plt.close(); print("Saved user_capacity.png")

# 5. Attacker capacity — 1 green curve
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
s = smooth(att_arr)
ax.plot(eps[:len(s)], s, color=C_ATT, linewidth=LINE_W, label='Attacker 0')
ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color=C_ATT)
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Attacker Eavesdropping Capacity (bits/s/Hz)', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('Attacker Eavesdropping Capacity per Episode (Averaged)', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(0, 1)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right', framealpha=0.9, edgecolor='#cccccc')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'attaker_capacity.png'), dpi=150, facecolor=BG); plt.close(); print("Saved attaker_capacity.png")

# 6. RIS distance
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
s = smooth(all_dists)
ax.plot(eps[:len(s)], s, color=C_USER1, linewidth=LINE_W)
ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color=C_USER1)
ax.axhline(y=0, color='#cccccc', linestyle='--', alpha=0.6)
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Distance (m)', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('UAV-RIS Distance (Smoothed)', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'ris_distance.png'), dpi=150, facecolor=BG); plt.close(); print("Saved ris_distance.png")

# 7. RIS scheduling
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
s = smooth(all_ris_skd)
ax.plot(eps[:len(s)], s, color='#7b1fa2', linewidth=LINE_W)
ax.fill_between(eps[:len(s)], s, alpha=FILL_ALPHA, color='#7b1fa2')
ax.set_xlabel('Episode', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_ylabel('Scheduling Rate', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
ax.set_title('RIS Element Scheduling Rate (Reflect vs Jam)', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
ax.set_ylim(-0.05, 1.05)
style_ax(ax)
plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'ris_scheduling.png'), dpi=150, facecolor=BG); plt.close(); print("Saved ris_scheduling.png")

# 8. RIS phase plots
f_last = os.path.join(path, 'simulation_result_ep_1000.mat')
if not os.path.exists(f_last):
    f_last = os.path.join(path, 'simulation_result_ep_999.mat')
if os.path.exists(f_last):
    data = scipy.io.loadmat(f_last)
    ep_key = 'result_1000' if 'result_1000' in data else 'result_999'
    result = data[ep_key]
    reflect_coeff = result['reflecting_coefficient'][0,0]
    n_elem = reflect_coeff.shape[1]
    for elem in range(n_elem):
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        phase = np.angle(reflect_coeff[:, elem])
        magnitude = np.abs(reflect_coeff[:, elem])
        ax.plot(phase, color='#7b1fa2', linewidth=1.5, label='Phase')
        ax2 = ax.twinx()
        ax2.plot(magnitude, color=C_USER0, linewidth=1, alpha=0.5, label='Magnitude')
        ax.set_xlabel('Step', fontsize=LABEL_SIZE, color=TICK_COLOR, fontweight='bold')
        ax.set_ylabel('Phase (rad)', fontsize=LABEL_SIZE, color='#7b1fa2', fontweight='bold')
        ax2.set_ylabel('Magnitude', fontsize=LABEL_SIZE, color=C_USER0, fontweight='bold')
        ax.set_title(f'RIS Element #{elem} Phase & Magnitude (v5)', fontsize=TITLE_SIZE, pad=12, color='#222222', fontweight='bold')
        style_ax(ax)
        ax2.tick_params(colors=TICK_COLOR, labelsize=TICK_SIZE)
        for sp in ax2.spines.values(): sp.set_color('#dddddd'); sp.set_linewidth(0.7)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=LEGEND_SIZE, framealpha=0.9, edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'RIS', f'RIS_{elem}_element.png'), dpi=150, facecolor=BG)
        plt.close(); print(f"Saved RIS_{elem}_element.png")

print("\nAll v5 plots generated!")
