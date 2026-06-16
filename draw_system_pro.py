#!/usr/bin/env python3
"""
专业级系统模型图 — 接近参考论文风格
使用 matplotlib 高级功能: 渐变、阴影、3D透视效果
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import patheffects

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 10), dpi=250)
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-1.5, 11.5)
ax.set_aspect('equal')
ax.axis('off')

# ═══════════════════════════════════════════
#  渐变背景 (天空)
# ═══════════════════════════════════════════
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack([gradient]*10)
ax.imshow(gradient, aspect='auto', cmap=plt.cm.Blues,
          extent=[-0.5, 15.5, 9.5, 11.5], alpha=0.15, zorder=0)

# ═══════════════════════════════════════════
#  地面 (带纹理)
# ═══════════════════════════════════════════
ground = FancyBboxPatch((-0.5, -1.5), 16, 2.0, boxstyle="square,pad=0",
                        facecolor='#D7CCC8', edgecolor='#8D6E63', linewidth=1.5, zorder=1)
ax.add_patch(ground)
# 地面纹理
for gx in np.arange(-0.3, 15.5, 0.6):
    ax.plot([gx, gx+0.25], [-0.1, -0.5], color='#BCAAA4', lw=0.4, alpha=0.5, zorder=1)

# ═══════════════════════════════════════════
#  建筑物 (3D透视效果)
# ═══════════════════════════════════════════
def draw_building(ax, x, y, w, h, color='#CFD8DC', edge='#90A4AE', depth=0.3):
    """画3D透视建筑物"""
    # 正面
    front = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                          facecolor=color, edgecolor=edge, linewidth=1, zorder=2)
    ax.add_patch(front)
    # 右侧面 (3D透视)
    side_x = [x+w, x+w+depth, x+w+depth, x+w]
    side_y = [y, y+depth*0.7, y+h+depth*0.7, y+h]
    side = Polygon(list(zip(side_x, side_y)), closed=True,
                   facecolor='#B0BEC5', edgecolor=edge, linewidth=0.8, zorder=2)
    ax.add_patch(side)
    # 顶面
    top_x = [x, x+depth, x+w+depth, x+w]
    top_y = [y+h, y+h+depth*0.7, y+h+depth*0.7, y+h]
    top = Polygon(list(zip(top_x, top_y)), closed=True,
                  facecolor='#ECEFF1', edgecolor=edge, linewidth=0.8, zorder=2)
    ax.add_patch(top)
    # 窗户
    for row in range(int(h//0.6)):
        for col in range(int(w//0.5)):
            wx = x + 0.15 + col * 0.5
            wy = y + 0.2 + row * 0.6
            if wy + 0.3 < y + h:
                win = Rectangle((wx, wy), 0.3, 0.35, facecolor='#B3E5FC',
                               edgecolor='#90A4AE', linewidth=0.3, alpha=0.8, zorder=3)
                ax.add_patch(win)

draw_building(ax, 0.3, 0.0, 1.5, 5.5)
draw_building(ax, 2.2, 0.0, 1.2, 4.0)
draw_building(ax, 12.5, 0.0, 1.8, 5.0)

# ═══════════════════════════════════════════
#  无人机 (带3D阴影)
# ═══════════════════════════════════════════
uav_x, uav_y = 7.5, 9.0

# 阴影
shadow = plt.Circle((uav_x+0.15, uav_y-0.15), 0.7, color='black', alpha=0.15, zorder=3)
ax.add_patch(shadow)

# 机身
body = FancyBboxPatch((uav_x-0.6, uav_y-0.25), 1.2, 0.5,
                      boxstyle="round,pad=0.08",
                      facecolor='#2C5AA0', edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(body)
ax.text(uav_x, uav_y, 'UAV', fontsize=11, ha='center', va='center',
        color='white', fontweight='bold', zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground='#1A3A6A')])

# 旋翼 (椭圆, 带运动模糊感)
for rx, ry, rot in [(-1.0, 0.5, 20), (1.0, 0.5, -20), (-1.0, -0.5, -20), (1.0, -0.5, 20)]:
    rotor = mpatches.Ellipse((uav_x+rx, uav_y+ry), 0.8, 0.2, angle=rot,
                             facecolor='#5C8BBF', edgecolor='#2C5AA0',
                             linewidth=0.8, alpha=0.7, zorder=4)
    ax.add_patch(rotor)
    # 旋翼臂
    ax.plot([uav_x+rx*0.5, uav_x+rx*0.85], [uav_y+ry*0.5, uav_y+ry*0.85],
            color='#2C5AA0', lw=1.8, alpha=0.6, zorder=4)

# ═══════════════════════════════════════════
#  FAS 天线阵列
# ═══════════════════════════════════════════
fas_y = uav_y - 0.6
fas_w = 2.8
fas_h = 0.25

# FAS 基板 (渐变效果)
fas_base = FancyBboxPatch((uav_x-fas_w/2, fas_y-fas_h/2), fas_w, fas_h,
                          boxstyle="round,pad=0.03",
                          facecolor='#1A8A5C', edgecolor='white', linewidth=1.2, zorder=5)
ax.add_patch(fas_base)
ax.text(uav_x, fas_y+fas_h/2+0.15, 'FAS ($N$=12端口)', fontsize=8, ha='center',
        va='bottom', color='#1A8A5C', fontweight='bold', zorder=6)

# 端口
n_ports = 12
port_spacing = fas_w / (n_ports + 1)
active_ports = [3, 7, 10]
for i in range(n_ports):
    px = uav_x - fas_w/2 + (i+1) * port_spacing
    active = (i+1) in active_ports
    color = '#4CAF50' if active else '#90A4AE'
    size = 0.1 if active else 0.07
    circle = plt.Circle((px, fas_y), size, color=color, ec='white',
                        lw=0.8 if not active else 1.2, zorder=6)
    ax.add_patch(circle)

# ═══════════════════════════════════════════
#  主动 RIS (建筑上的面板)
# ═══════════════════════════════════════════
ris_x, ris_y = 7.5, 4.5
ris_w, ris_h = 2.5, 0.7

# RIS 面板
ris_panel = FancyBboxPatch((ris_x-ris_w/2, ris_y-ris_h/2), ris_w, ris_h,
                           boxstyle="round,pad=0.04",
                           facecolor='#D4760A', edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(ris_panel)

# RIS 单元格 (网格效果)
for row in range(2):
    for col in range(8):
        cx = ris_x - ris_w/2 + 0.12 + col * (ris_w-0.24)/8
        cy = ris_y - ris_h/2 + 0.08 + row * (ris_h-0.16)/2
        cell = Rectangle((cx, cy), (ris_w-0.3)/8-0.02, (ris_h-0.2)/2-0.02,
                         facecolor='#FFB74D', edgecolor='#D4760A',
                         linewidth=0.3, alpha=0.8, zorder=6)
        ax.add_patch(cell)

ax.text(ris_x, ris_y+ris_h/2+0.15, '主动 RIS ($M$=64 单元)', fontsize=9,
        ha='center', va='bottom', color='#D4760A', fontweight='bold', zorder=6)
ax.text(ris_x, ris_y-ris_h/2-0.1, '信号反射 + 人工噪声', fontsize=7,
        ha='center', va='top', color='#BF360C', zorder=6)

# ═══════════════════════════════════════════
#  合法用户 (手机图标)
# ═══════════════════════════════════════════
def draw_phone(ax, x, y, label, color='#2196F3'):
    """画手机形状的用户图标"""
    # 手机阴影
    phone_shadow = FancyBboxPatch((x-0.18+0.05, y-0.3+0.05), 0.36, 0.6,
                                  boxstyle="round,pad=0.03",
                                  facecolor='black', alpha=0.15, zorder=3)
    ax.add_patch(phone_shadow)
    # 手机主体
    phone = FancyBboxPatch((x-0.18, y-0.3), 0.36, 0.6,
                           boxstyle="round,pad=0.03",
                           facecolor=color, edgecolor='white', linewidth=1.2, zorder=5)
    ax.add_patch(phone)
    # 屏幕
    screen = FancyBboxPatch((x-0.14, y-0.22), 0.28, 0.4,
                            boxstyle="round,pad=0.02",
                            facecolor='white', edgecolor=color, linewidth=0.5, alpha=0.9, zorder=6)
    ax.add_patch(screen)
    # 手机顶部听筒
    ax.plot([x-0.05, x+0.05], [y+0.26, y+0.26], color=color, lw=1.5, zorder=6)
    # 锁图标 (用文字模拟)
    ax.text(x, y-0.05, '🔒', fontsize=8, ha='center', va='center', zorder=7)
    ax.plot([x-0.06, x+0.06], [y+0.02, y+0.02], color=color, lw=1.5, zorder=7)
    ax.plot([x-0.06, x+0.06], [y-0.12, y-0.12], color=color, lw=1.5, zorder=7)
    # 标签
    ax.text(x, y-0.45, label, fontsize=9, ha='center', va='top',
            color='#1565C0', fontweight='bold', zorder=6)

draw_phone(ax, 2.5, 1.5, 'User 1')
draw_phone(ax, 12.5, 1.5, 'User 2')

# ═══════════════════════════════════════════
#  窃听者 (耳机图标)
# ═══════════════════════════════════════════
eve_x, eve_y = 7.5, 1.5

# 阴影
eve_shadow = plt.Circle((eve_x+0.08, eve_y-0.08), 0.35, color='black', alpha=0.15, zorder=3)
ax.add_patch(eve_shadow)

# 头部
eve_head = plt.Circle((eve_x, eve_y), 0.35, facecolor='#E53935', edgecolor='white',
                      linewidth=1.5, zorder=5)
ax.add_patch(eve_head)
ax.text(eve_x, eve_y, 'E', fontsize=12, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)

# 耳机
headphone_left = mpatches.Arc((eve_x-0.32, eve_y+0.1), 0.2, 0.35, angle=0,
                              theta1=90, theta2=270, color='#333', lw=2.5, zorder=5)
headphone_right = mpatches.Arc((eve_x+0.32, eve_y+0.1), 0.2, 0.35, angle=0,
                               theta1=270, theta2=90, color='#333', lw=2.5, zorder=5)
headphone_top = mpatches.Arc((eve_x, eve_y+0.25), 0.55, 0.2, angle=0,
                             theta1=0, theta2=180, color='#333', lw=2.5, zorder=5)
ax.add_patch(headphone_left)
ax.add_patch(headphone_right)
ax.add_patch(headphone_top)

ax.text(eve_x, eve_y-0.5, 'Eavesdropper $p$', fontsize=8, ha='center', va='top',
        color='#B71C1C', fontweight='bold', zorder=6)

# ═══════════════════════════════════════════
#  信号路径 (专业箭头样式)
# ═══════════════════════════════════════════

signal_base = dict(arrowstyle='->', color='#2196F3', lw=1.8, zorder=4)
jam_base = dict(arrowstyle='->', color='#E53935', lw=1.8, linestyle='dashed', zorder=4)

# ── FAS → U1 直射 ──
ax.annotate('', xy=(2.5, 1.8), xytext=(uav_x-1.0, fas_y-0.2),
            arrowprops=dict(**signal_base, connectionstyle='arc3,rad=0.2'))
ax.text(3.5, 5.2, '$\\mathbf{h}^{U}_{U,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ── FAS → U2 直射 ──
ax.annotate('', xy=(12.5, 1.8), xytext=(uav_x+1.0, fas_y-0.2),
            arrowprops=dict(**signal_base, connectionstyle='arc3,rad=-0.2'))
ax.text(11.5, 5.2, '$\\mathbf{h}^{U}_{U,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ── FAS → E 直射 (窃听, 红色虚线) ──
ax.annotate('', xy=(eve_x, eve_y+0.4), xytext=(uav_x, fas_y-0.2),
            arrowprops=dict(**jam_base))
ax.text(8.0, 5.5, '$\\mathbf{h}^{U}_{U,p}$', fontsize=9, color='#E53935',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ── FAS → RIS (上行) ──
ax.annotate('', xy=(ris_x-0.8, ris_y+ris_h/2), xytext=(uav_x-0.3, fas_y-0.2),
            arrowprops=dict(**signal_base, connectionstyle='arc3,rad=0.05'))
ax.text(6.3, 6.8, '$\\mathbf{h}^{U}_{U,R}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ── RIS → U1 (反射) ──
ax.annotate('', xy=(2.5, 1.8), xytext=(ris_x-1.2, ris_y+ris_h/2),
            arrowprops=dict(**signal_base, connectionstyle='arc3,rad=0.15'))
ax.text(3.2, 3.2, '$\\mathbf{h}^{U}_{R,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ── RIS → U2 (反射) ──
ax.annotate('', xy=(12.5, 1.8), xytext=(ris_x+1.2, ris_y+ris_h/2),
            arrowprops=dict(**signal_base, connectionstyle='arc3,rad=-0.15'))
ax.text(11.8, 3.2, '$\\mathbf{h}^{U}_{R,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ── RIS → E (干扰) ──
ax.annotate('', xy=(eve_x-0.1, eve_y+0.4), xytext=(ris_x, ris_y-ris_h/2),
            arrowprops=dict(**jam_base))
ax.text(7.0, 3.0, '$\\mathbf{h}^{U}_{R,p}$', fontsize=9, color='#E53935',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1, boxstyle='round,pad=0.2'))

# ═══════════════════════════════════════════
#  DRL 框图 (顶部)
# ═══════════════════════════════════════════
drl_x, drl_y = 7.5, 10.8
drl_w, drl_h = 5.0, 0.8

# DRL 外框
drl_box = FancyBboxPatch((drl_x-drl_w/2, drl_y-drl_h/2), drl_w, drl_h,
                         boxstyle="round,pad=0.08",
                         facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1.2, zorder=5)
ax.add_patch(drl_box)
ax.text(drl_x, drl_y+drl_h/2-0.12, 'Dual-Agent Twin-TD3', fontsize=10,
        ha='center', va='top', fontweight='bold', zorder=6)

# Agent 1 框
a1_box = FancyBboxPatch((drl_x-2.2, drl_y-0.35), 1.6, 0.4,
                        boxstyle="round,pad=0.05",
                        facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=1, zorder=6)
ax.add_patch(a1_box)
ax.text(drl_x-1.4, drl_y-0.15, 'Agent 1\n(FAS+RIS)', fontsize=7,
        ha='center', va='center', color='#1565C0', zorder=7)

# Agent 2 框
a2_box = FancyBboxPatch((drl_x+0.6, drl_y-0.35), 1.6, 0.4,
                        boxstyle="round,pad=0.05",
                        facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=1, zorder=6)
ax.add_patch(a2_box)
ax.text(drl_x+1.4, drl_y-0.15, 'Agent 2\n(UAV)', fontsize=7,
        ha='center', va='center', color='#2E7D32', zorder=7)

# ═══════════════════════════════════════════
#  高度标注
# ═══════════════════════════════════════════
ax.annotate('', xy=(0.1, 9.2), xytext=(0.1, 0.5),
            arrowprops=dict(arrowstyle='<->', color='#999', lw=1, linestyle='--'))
ax.text(0.1, 4.8, '$H$=50m', fontsize=8, ha='center', va='center',
        color='#666', rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))

# ═══════════════════════════════════════════
#  图例
# ═══════════════════════════════════════════
legend_items = [
    ('#2196F3', '-', '信号链路 (LOS)'),
    ('#E53935', '--', '窃听/干扰链路'),
    ('#4CAF50', 'o', 'FAS 激活端口'),
    ('#90A4AE', 'o', 'FAS 未激活端口'),
]
for i, (color, ls, label) in enumerate(legend_items):
    ly = 10.5 - i * 0.4
    if ls == 'o':
        ax.plot(14.0, ly, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
    elif ls == '-':
        ax.plot([13.7, 14.3], [ly, ly], color=color, lw=2)
    else:
        ax.plot([13.7, 14.3], [ly, ly], color=color, lw=2, linestyle='--')
    ax.text(14.5, ly, label, fontsize=7, va='center', color='#333')

# ═══════════════════════════════════════════
#  标题
# ═══════════════════════════════════════════
ax.text(7.5, 11.2, 'FAS辅助主动RIS无人机安全通信系统模型',
        fontsize=14, ha='center', va='center', fontweight='bold', color='#1a1a1a')

# 参数标注
params = '$f_c$=28GHz | FAS: $N$=12 | RIS: $M$=64 | $K$=2 | $P$=1 | $H$=50m'
ax.text(7.5, 10.0, params, fontsize=7.5, ha='center', va='center', color='#666',
        bbox=dict(fc='#f5f5f5', ec='#ccc', alpha=0.9, boxstyle='round,pad=0.3', lw=0.8))

# ═══════════════════════════════════════════
#  区域标注
# ═══════════════════════════════════════════
# 信号增强区
zone1 = FancyBboxPatch((1.2, 0.3), 3.0, 2.0, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#2196F3', linewidth=0.8,
                       linestyle='--', alpha=0.4, zorder=2)
ax.add_patch(zone1)
ax.text(2.7, 2.5, '信号增强区', fontsize=7, ha='center', color='#2196F3', alpha=0.6, zorder=3)

# 干扰抑制区
zone2 = FancyBboxPatch((6.2, 0.3), 2.8, 2.0, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#E53935', linewidth=0.8,
                       linestyle='--', alpha=0.4, zorder=2)
ax.add_patch(zone2)
ax.text(7.6, 2.5, '干扰抑制区', fontsize=7, ha='center', color='#E53935', alpha=0.6, zorder=3)

# ═══════════════════════════════════════════
#  保存
# ═══════════════════════════════════════════
output_dir = r"C:\Users\红\Desktop\0606强化学习用于保密能源——流体天线辅助无人机保密通信"
fig.savefig(f"{output_dir}/系统模型_v3.pdf", bbox_inches='tight', pad_inches=0.3, dpi=300)
fig.savefig(f"{output_dir}/系统模型_v3.png", bbox_inches='tight', pad_inches=0.3, dpi=300)
print(f"PDF: {output_dir}/系统模型_v3.pdf")
print(f"PNG: {output_dir}/系统模型_v3.png")
plt.close()
