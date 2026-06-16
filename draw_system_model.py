#!/usr/bin/env python3
"""
UAV-FAS-RIS 安全通信系统模型图
输出: 系统模型.pdf (矢量图，可直接插入论文)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
import numpy as np

# ── 中文字体 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 9), dpi=200)
ax.set_xlim(-1.5, 13.5)
ax.set_ylim(-1.0, 10.5)
ax.set_aspect('equal')
ax.axis('off')

# ═══════════════════════════════════════════
#  颜色定义
# ═══════════════════════════════════════════
C_UAV      = '#2c5aa0'   # 无人机 深蓝
C_FAS      = '#1a8a5c'   # FAS 绿
C_RIS      = '#d4760a'   # RIS 橙
C_USER     = '#2196F3'   # 用户 蓝
C_EVE      = '#e53935'   # 窃听者 红
C_SIGNAL   = '#2196F3'   # 信号线 蓝
C_JAM      = '#e53935'   # 干扰线 红
C_GROUND   = '#8d6e63'   # 地面 棕
C_GRID     = '#e0e0e0'   # 网格浅灰

# ═══════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════
def draw_dashed_arrow(ax, start, end, color, style='dashed', lw=1.5, alpha=0.8):
    """画虚线箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color,
                                linestyle=style, lw=lw, alpha=alpha,
                                connectionstyle='arc3,rad=0'))

def draw_signal_path(ax, start, end, color, lw=1.8, alpha=0.7, rad=0.0, style='-'):
    """画信号路径（带弧度）"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color,
                                linestyle=style, lw=lw, alpha=alpha,
                                connectionstyle=f'arc3,rad={rad}'))

def draw_entity(ax, x, y, w, h, color, label, fontsize=10, icon=None):
    """画实体框"""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white',
                          linewidth=2, alpha=0.9, zorder=5)
    ax.add_patch(rect)
    if icon:
        ax.text(x, y + 0.15, icon, fontsize=fontsize+8, ha='center', va='center',
                color='white', fontweight='bold', zorder=6)
        ax.text(x, y - 0.35, label, fontsize=fontsize-1, ha='center', va='center',
                color='white', fontweight='normal', zorder=6)
    else:
        ax.text(x, y, label, fontsize=fontsize, ha='center', va='center',
                color='white', fontweight='bold', zorder=6)

def draw_antenna_port(ax, x, y, idx, active=False):
    """画FAS天线端口"""
    color = '#4CAF50' if active else '#90A4AE'
    size = 0.18 if active else 0.12
    circle = plt.Circle((x, y), size, color=color, ec='white', lw=1, zorder=7)
    ax.add_patch(circle)
    if active:
        ax.text(x, y, str(idx), fontsize=5, ha='center', va='center',
                color='white', fontweight='bold', zorder=8)

# ═══════════════════════════════════════════
#  地面
# ═══════════════════════════════════════════
ground = plt.Polygon([(-1, 0.3), (13, 0.3), (13, -0.8), (-1, -0.8)],
                     closed=True, fc='#d7ccc8', ec=C_GROUND, lw=1.5, zorder=1)
ax.add_patch(ground)
for gx in np.arange(-0.5, 13, 0.8):
    ax.plot([gx, gx+0.3], [-0.3, -0.6], color=C_GROUND, lw=0.5, alpha=0.4, zorder=1)

# ═══════════════════════════════════════════
#  坐标系标注
# ═══════════════════════════════════════════
# H = 50m 标注
ax.annotate('', xy=(0.2, 7.0), xytext=(0.2, 0.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1.2, linestyle='--'))
ax.text(0.2, 3.75, 'H=50m', fontsize=8, ha='center', va='center',
        color='#666', rotation=90, bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

# ═══════════════════════════════════════════
#  1. 无人机 (UAV + FAS)
# ═══════════════════════════════════════════
uav_x, uav_y = 6.5, 8.0

# 无人机机身
uav_body = plt.Circle((uav_x, uav_y), 0.6, color=C_UAV, ec='white', lw=2, zorder=5)
ax.add_patch(uav_body)
ax.text(uav_x, uav_y, 'UAV', fontsize=11, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)

# 旋翼 (4个)
rotor_angles = [45, 135, 225, 315]
for angle in rotor_angles:
    rad = np.radians(angle)
    rx = uav_x + 1.2 * np.cos(rad)
    ry = uav_y + 0.8 * np.sin(rad)
    rotor = plt.Circle((rx, ry), 0.25, color=C_UAV, ec='white', lw=1.5, alpha=0.7, zorder=4)
    ax.add_patch(rotor)
    # 旋翼臂
    ax.plot([uav_x + 0.5*np.cos(rad), rx - 0.2*np.cos(rad)],
            [uav_y + 0.35*np.sin(rad), ry - 0.1*np.sin(rad)],
            color=C_UAV, lw=2, alpha=0.6, zorder=4)

# FAS 天线阵列 (无人机底部)
fas_y = uav_y - 0.8
n_ports = 12
port_spacing = 0.22
fas_start_x = uav_x - (n_ports - 1) * port_spacing / 2

# FAS 基板
fas_base = FancyBboxPatch((fas_start_x - 0.2, fas_y - 0.15),
                          (n_ports - 1) * port_spacing + 0.4, 0.3,
                          boxstyle="round,pad=0.05",
                          facecolor=C_FAS, edgecolor='white', lw=1.5, alpha=0.85, zorder=5)
ax.add_patch(fas_base)
ax.text(uav_x, fas_y + 0.35, 'FAS (N=12端口)', fontsize=8, ha='center', va='bottom',
        color=C_FAS, fontweight='bold', zorder=6)

# 激活端口 (高亮)
active_ports = [3, 7, 10]  # 假设激活端口 3,7,10
for i in range(n_ports):
    px = fas_start_x + i * port_spacing
    draw_antenna_port(ax, px, fas_y, i+1, active=(i+1) in active_ports)

# ═══════════════════════════════════════════
#  2. 主动 RIS
# ═══════════════════════════════════════════
ris_x, ris_y = 6.5, 2.2
ris_w, ris_h = 3.0, 0.8

# RIS 主体
ris_rect = FancyBboxPatch((ris_x - ris_w/2, ris_y - ris_h/2), ris_w, ris_h,
                          boxstyle="round,pad=0.08",
                          facecolor=C_RIS, edgecolor='white', lw=2, alpha=0.9, zorder=5)
ax.add_patch(ris_rect)

# RIS 单元格 (8x8网格示意)
for row in range(3):
    for col in range(10):
        cx = ris_x - ris_w/2 + 0.15 + col * (ris_w - 0.3) / 10
        cy = ris_y - ris_h/2 + 0.12 + row * (ris_h - 0.24) / 3
        cell = plt.Rectangle((cx, cy), 0.22, 0.16,
                             fc='white', ec=C_RIS, lw=0.5, alpha=0.6, zorder=6)
        ax.add_patch(cell)

ax.text(ris_x, ris_y + ris_h/2 + 0.25, '主动 RIS (M=64 单元)', fontsize=9,
        ha='center', va='bottom', color=C_RIS, fontweight='bold', zorder=6)
ax.text(ris_x, ris_y, '信号反射 + 人工噪声', fontsize=7,
        ha='center', va='center', color='white', fontweight='normal', zorder=6)

# ═══════════════════════════════════════════
#  3. 合法用户
# ═══════════════════════════════════════════
user1_x, user1_y = 3.0, 1.0
user2_x, user2_y = 10.0, 1.0

# 用户1
user1_icon = plt.Circle((user1_x, user1_y), 0.35, color=C_USER, ec='white', lw=2, zorder=5)
ax.add_patch(user1_icon)
ax.text(user1_x, user1_y, 'U1', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)
ax.text(user1_x, user1_y - 0.55, '合法用户1', fontsize=8, ha='center', va='top',
        color=C_USER, fontweight='bold', zorder=6)

# 用户2
user2_icon = plt.Circle((user2_x, user2_y), 0.35, color=C_USER, ec='white', lw=2, zorder=5)
ax.add_patch(user2_icon)
ax.text(user2_x, user2_y, 'U2', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)
ax.text(user2_x, user2_y - 0.55, '合法用户2', fontsize=8, ha='center', va='top',
        color=C_USER, fontweight='bold', zorder=6)

# ═══════════════════════════════════════════
#  4. 窃听者
# ═══════════════════════════════════════════
eve_x, eve_y = 8.5, 1.0

eve_icon = plt.Circle((eve_x, eve_y), 0.35, color=C_EVE, ec='white', lw=2, zorder=5)
ax.add_patch(eve_icon)
ax.text(eve_x, eve_y, 'E', fontsize=11, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)
ax.text(eve_x, eve_y - 0.55, '窃听者', fontsize=8, ha='center', va='top',
        color=C_EVE, fontweight='bold', zorder=6)

# ═══════════════════════════════════════════
#  5. 信号路径
# ═══════════════════════════════════════════

# ── 直射链路 (FAS → 用户) ──
draw_signal_path(ax, (fas_start_x + 2*port_spacing, fas_y - 0.2),
                 (user1_x, user1_y + 0.4), C_SIGNAL, lw=2.0, rad=-0.15)
draw_signal_path(ax, (fas_start_x + 7*port_spacing, fas_y - 0.2),
                 (user2_x, user2_y + 0.4), C_SIGNAL, lw=2.0, rad=0.15)

# 直射链路标注
ax.text(3.8, 5.0, '直射链路', fontsize=7, color=C_SIGNAL, ha='center',
        rotation=-38, fontweight='bold',
        bbox=dict(fc='white', ec=C_SIGNAL, alpha=0.8, boxstyle='round,pad=0.2', lw=0.8))

# ── 直射链路 → 窃听者 ──
draw_signal_path(ax, (fas_start_x + 9*port_spacing, fas_y - 0.2),
                 (eve_x, eve_y + 0.4), C_EVE, lw=1.5, rad=0.0, style='dashed')
ax.text(8.3, 5.0, '直射链路', fontsize=7, color=C_EVE, ha='center',
        rotation=-35, fontweight='bold',
        bbox=dict(fc='white', ec=C_EVE, alpha=0.8, boxstyle='round,pad=0.2', lw=0.8))

# ── FAS → RIS (上行) ──
draw_signal_path(ax, (uav_x, uav_y - 0.9), (ris_x - 0.8, ris_y + ris_h/2),
                 C_SIGNAL, lw=2.0, rad=-0.1)
ax.text(uav_x - 1.5, (uav_y + ris_y)/2 + 0.3, 'FAS→RIS', fontsize=7,
        color=C_SIGNAL, ha='center', fontweight='bold', rotation=-82,
        bbox=dict(fc='white', ec=C_SIGNAL, alpha=0.8, boxstyle='round,pad=0.15', lw=0.8))

# ── RIS → 用户 (反射链路，相长干涉) ──
draw_signal_path(ax, (ris_x - 1.0, ris_y + ris_h/2),
                 (user1_x + 0.2, user1_y + 0.4), C_SIGNAL, lw=2.0, rad=0.15)
draw_signal_path(ax, (ris_x + 1.0, ris_y + ris_h/2),
                 (user2_x - 0.2, user2_y + 0.4), C_SIGNAL, lw=2.0, rad=-0.15)

# 反射链路标注
ax.text(3.5, 3.0, 'RIS反射链路\n(相长干涉)', fontsize=7, color=C_SIGNAL, ha='center',
        fontweight='bold',
        bbox=dict(fc='white', ec=C_SIGNAL, alpha=0.8, boxstyle='round,pad=0.2', lw=0.8))

# ── RIS → 窃听者 (干扰链路，人工噪声) ──
draw_signal_path(ax, (ris_x + 1.2, ris_y + ris_h/2),
                 (eve_x - 0.2, eve_y + 0.4), C_JAM, lw=2.0, rad=-0.2, style='dashed')

# 干扰标注
ax.text(8.2, 3.2, 'RIS干扰\n(人工噪声)', fontsize=7, color=C_JAM, ha='center',
        fontweight='bold',
        bbox=dict(fc='white', ec=C_JAM, alpha=0.8, boxstyle='round,pad=0.2', lw=0.8))

# ═══════════════════════════════════════════
#  6. 图例
# ═══════════════════════════════════════════
legend_x, legend_y = 11.5, 9.0
legend_items = [
    (C_SIGNAL, '-', '信号链路'),
    (C_JAM, '--', '干扰链路'),
    ('#4CAF50', 'o', '激活端口'),
    ('#90A4AE', 'o', '未激活端口'),
]
for i, (color, marker, label) in enumerate(legend_items):
    ly = legend_y - i * 0.5
    if marker == 'o':
        ax.plot(legend_x, ly, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=1)
    elif marker == '-':
        ax.plot([legend_x - 0.3, legend_x + 0.3], [ly, ly], color=color, lw=2)
    else:
        ax.plot([legend_x - 0.3, legend_x + 0.3], [ly, ly], color=color, lw=2, linestyle='--')
    ax.text(legend_x + 0.5, ly, label, fontsize=7, va='center', color='#333')

# ═══════════════════════════════════════════
#  7. 标题和系统参数标注
# ═══════════════════════════════════════════
ax.text(6.5, 10.2, 'FAS辅助主动RIS无人机安全通信系统模型',
        fontsize=14, ha='center', va='center', fontweight='bold', color='#1a1a1a')

# 参数小字
params_text = ('f_c=28GHz  |  FAS: N=12端口  |  RIS: M=64单元  |  '
               'K=2用户  |  P=1窃听者  |  H=50m')
ax.text(6.5, 9.6, params_text, fontsize=7, ha='center', va='center',
        color='#666',
        bbox=dict(fc='#f5f5f5', ec='#ccc', alpha=0.9, boxstyle='round,pad=0.3', lw=0.8))

# ═══════════════════════════════════════════
#  8. 信号方向标注箭头 (装饰)
# ═══════════════════════════════════════════
# 虚线框标注 "信号增强区"
box_user = mpatches.FancyBboxPatch((1.5, 0.3), 3.5, 1.8,
                                    boxstyle="round,pad=0.1",
                                    fc='none', ec=C_SIGNAL, lw=1, ls='--', alpha=0.4)
ax.add_patch(box_user)
ax.text(3.25, 2.2, '信号增强区', fontsize=6, ha='center', color=C_SIGNAL, alpha=0.6)

box_eve = mpatches.FancyBboxPatch((7.5, 0.3), 2.5, 1.8,
                                    boxstyle="round,pad=0.1",
                                    fc='none', ec=C_EVE, lw=1, ls='--', alpha=0.4)
ax.add_patch(box_eve)
ax.text(8.75, 2.2, '干扰抑制区', fontsize=6, ha='center', color=C_EVE, alpha=0.6)

# ═══════════════════════════════════════════
#  保存
# ═══════════════════════════════════════════
output_dir = r"C:\Users\红\Desktop\0606强化学习用于保密能源——流体天线辅助无人机保密通信"
fig.savefig(f"{output_dir}/系统模型.pdf", bbox_inches='tight', pad_inches=0.2)
fig.savefig(f"{output_dir}/系统模型.png", bbox_inches='tight', pad_inches=0.2, dpi=300)
print("系统模型图已保存:")
print(f"  PDF: {output_dir}/系统模型.pdf")
print(f"  PNG: {output_dir}/系统模型.png")
plt.close()
