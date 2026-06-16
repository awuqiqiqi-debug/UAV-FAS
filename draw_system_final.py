#!/usr/bin/env python3
"""
系统模型图 最终版
- RIS 在12.5m建筑上
- 用户/窃听者在地面，简单手机形状
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 10), dpi=250)
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-1.5, 11.5)
ax.set_aspect('equal')
ax.axis('off')

# ═══════════════════════════════════════════
#  背景
# ═══════════════════════════════════════════
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack([gradient]*10)
ax.imshow(gradient, aspect='auto', cmap=plt.cm.Blues,
          extent=[-0.5, 15.5, 9.5, 11.5], alpha=0.15, zorder=0)

# 地面
ground = FancyBboxPatch((-0.5, -1.5), 16, 2.0, boxstyle="square,pad=0",
                        facecolor='#D7CCC8', edgecolor='#8D6E63', linewidth=1.5, zorder=1)
ax.add_patch(ground)
for gx in np.arange(-0.3, 15.5, 0.6):
    ax.plot([gx, gx+0.25], [-0.1, -0.5], color='#BCAAA4', lw=0.4, alpha=0.5, zorder=1)

# ═══════════════════════════════════════════
#  建筑物 (RIS在12.5m建筑上)
# ═══════════════════════════════════════════

# 中间建筑 (RIS安装在上面, 高12.5m对应图上约4.5个单位)
ris_bld_x, ris_bld_y = 6.0, 0.0
ris_bld_w, ris_bld_h = 3.0, 5.0

# 建筑正面
bld_main = FancyBboxPatch((ris_bld_x, ris_bld_y), ris_bld_w, ris_bld_h,
                          boxstyle="square,pad=0",
                          facecolor='#B0BEC5', edgecolor='#78909C', linewidth=1.2, zorder=2)
ax.add_patch(bld_main)
# 3D侧面
side_x = [ris_bld_x+ris_bld_w, ris_bld_x+ris_bld_w+0.4, ris_bld_x+ris_bld_w+0.4, ris_bld_x+ris_bld_w]
side_y = [ris_bld_y, ris_bld_y+0.3, ris_bld_y+ris_bld_h+0.3, ris_bld_y+ris_bld_h]
side = Polygon(list(zip(side_x, side_y)), closed=True,
               facecolor='#90A4AE', edgecolor='#78909C', linewidth=0.8, zorder=2)
ax.add_patch(side)
# 顶面
top_x = [ris_bld_x, ris_bld_x+0.4, ris_bld_x+ris_bld_w+0.4, ris_bld_x+ris_bld_w]
top_y = [ris_bld_y+ris_bld_h, ris_bld_y+ris_bld_h+0.3, ris_bld_y+ris_bld_h+0.3, ris_bld_y+ris_bld_h]
top = Polygon(list(zip(top_x, top_y)), closed=True,
              facecolor='#CFD8DC', edgecolor='#78909C', linewidth=0.8, zorder=2)
ax.add_patch(top)
# 窗户
for row in range(6):
    for col in range(4):
        wx = ris_bld_x + 0.2 + col * 0.7
        wy = ris_bld_y + 0.2 + row * 0.8
        if wy + 0.5 < ris_bld_y + ris_bld_h:
            win = Rectangle((wx, wy), 0.45, 0.55, facecolor='#B3E5FC',
                           edgecolor='#90A4AE', linewidth=0.3, alpha=0.7, zorder=3)
            ax.add_patch(win)

# 左侧建筑
bld_left = FancyBboxPatch((0.3, 0.0), 1.5, 4.5, boxstyle="square,pad=0",
                          facecolor='#CFD8DC', edgecolor='#90A4AE', linewidth=1, zorder=2)
ax.add_patch(bld_left)
for row in range(5):
    for col in range(3):
        win = Rectangle((0.5+col*0.45, 0.2+row*0.85), 0.3, 0.5,
                        facecolor='#B3E5FC', edgecolor='#90A4AE', linewidth=0.3, alpha=0.7, zorder=3)
        ax.add_patch(win)

# 右侧建筑
bld_right = FancyBboxPatch((13.0, 0.0), 1.8, 5.2, boxstyle="square,pad=0",
                           facecolor='#CFD8DC', edgecolor='#90A4AE', linewidth=1, zorder=2)
ax.add_patch(bld_right)
for row in range(6):
    for col in range(3):
        win = Rectangle((13.2+col*0.5, 0.2+row*0.85), 0.35, 0.5,
                        facecolor='#B3E5FC', edgecolor='#90A4AE', linewidth=0.3, alpha=0.7, zorder=3)
        ax.add_patch(win)

# ═══════════════════════════════════════════
#  无人机 (UAV)
# ═══════════════════════════════════════════
uav_x, uav_y = 7.5, 9.0

# 阴影
shadow = plt.Circle((uav_x+0.15, uav_y-0.15), 0.7, color='black', alpha=0.12, zorder=3)
ax.add_patch(shadow)

# 旋翼
for rx, ry, rot in [(-1.0, 0.5, 20), (1.0, 0.5, -20), (-1.0, -0.5, -20), (1.0, -0.5, 20)]:
    rotor = mpatches.Ellipse((uav_x+rx, uav_y+ry), 0.8, 0.2, angle=rot,
                             facecolor='#5C8BBF', edgecolor='#2C5AA0',
                             linewidth=0.8, alpha=0.65, zorder=4)
    ax.add_patch(rotor)
    ax.plot([uav_x+rx*0.5, uav_x+rx*0.85], [uav_y+ry*0.5, uav_y+ry*0.85],
            color='#2C5AA0', lw=1.5, alpha=0.5, zorder=4)

# 机身
body = FancyBboxPatch((uav_x-0.6, uav_y-0.25), 1.2, 0.5,
                      boxstyle="round,pad=0.08",
                      facecolor='#2C5AA0', edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(body)
ax.text(uav_x, uav_y, 'UAV', fontsize=11, ha='center', va='center',
        color='white', fontweight='bold', zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground='#1A3A6A')])

# ═══════════════════════════════════════════
#  FAS 天线阵列
# ═══════════════════════════════════════════
fas_y = uav_y - 0.6
fas_w = 2.8
fas_h = 0.25

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
#  主动 RIS (在中间建筑顶部, 12.5m)
# ═══════════════════════════════════════════
ris_x = ris_bld_x + ris_bld_w / 2  # 建筑中心 x=7.5
ris_y = ris_bld_y + ris_bld_h + 0.3  # 建筑顶部 + 顶面偏移
ris_w, ris_h = 2.2, 0.55

# RIS 面板
ris_panel = FancyBboxPatch((ris_x-ris_w/2, ris_y-ris_h/2), ris_w, ris_h,
                           boxstyle="round,pad=0.03",
                           facecolor='#D4760A', edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(ris_panel)

# RIS 单元格
for row in range(2):
    for col in range(7):
        cx = ris_x - ris_w/2 + 0.1 + col * (ris_w-0.2)/7
        cy = ris_y - ris_h/2 + 0.06 + row * (ris_h-0.12)/2
        cell = Rectangle((cx, cy), (ris_w-0.25)/7-0.02, (ris_h-0.15)/2-0.02,
                         facecolor='#FFB74D', edgecolor='#D4760A',
                         linewidth=0.3, alpha=0.8, zorder=6)
        ax.add_patch(cell)

ax.text(ris_x, ris_y+ris_h/2+0.15, '主动 RIS ($M$=64 单元)', fontsize=9,
        ha='center', va='bottom', color='#D4760A', fontweight='bold', zorder=6)
ax.text(ris_x, ris_y-ris_h/2-0.08, '信号反射 + 人工噪声', fontsize=7,
        ha='center', va='top', color='#BF360C', zorder=6)

# ═══════════════════════════════════════════
#  用户和窃听者 (地面, 简单手机形状)
# ═══════════════════════════════════════════

def draw_phone_simple(ax, x, y, label, color='#2196F3'):
    """简单手机形状"""
    # 手机外框
    phone = FancyBboxPatch((x-0.15, y-0.25), 0.3, 0.5,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='white', linewidth=1.2, zorder=5)
    ax.add_patch(phone)
    # 屏幕
    screen = Rectangle((x-0.11, y-0.18), 0.22, 0.34,
                       facecolor='white', edgecolor=color, linewidth=0.5, alpha=0.9, zorder=6)
    ax.add_patch(screen)
    # 信号图标
    ax.plot([x, x], [y+0.08, y+0.15], color=color, lw=1.5, zorder=7)
    ax.plot([x-0.04, x+0.04], [y+0.15, y+0.15], color=color, lw=1.2, zorder=7)
    ax.plot([x-0.07, x+0.07], [y+0.19, y+0.19], color=color, lw=1, zorder=7)
    # 标签
    ax.text(x, y-0.35, label, fontsize=8, ha='center', va='top',
            color='#1565C0', fontweight='bold', zorder=6)

# 用户1 (地面, 左侧)
draw_phone_simple(ax, 2.5, 0.8, 'User 1')

# 用户2 (地面, 右侧)
draw_phone_simple(ax, 12.5, 0.8, 'User 2')

# 窃听者 (地面, 中间)
eve_x, eve_y = 7.5, 0.8

# 手机外框 (红色)
eve_phone = FancyBboxPatch((eve_x-0.15, eve_y-0.25), 0.3, 0.5,
                           boxstyle="round,pad=0.02",
                           facecolor='#E53935', edgecolor='white', linewidth=1.2, zorder=5)
ax.add_patch(eve_phone)
eve_screen = Rectangle((eve_x-0.11, eve_y-0.18), 0.22, 0.34,
                       facecolor='white', edgecolor='#E53935', linewidth=0.5, alpha=0.9, zorder=6)
ax.add_patch(eve_screen)
# 窃听图标 (眼睛)
ax.plot([eve_x-0.06, eve_x+0.06], [eve_y+0.05, eve_y+0.05], color='#E53935', lw=1.5, zorder=7)
ax.plot([eve_x-0.03, eve_x+0.03], [eve_y+0.05, eve_y+0.05], color='#E53935', lw=3, zorder=7)
ax.text(eve_x, eve_y-0.35, 'Eavesdropper $p$', fontsize=8, ha='center', va='top',
        color='#B71C1C', fontweight='bold', zorder=6)

# ═══════════════════════════════════════════
#  信号路径
# ═══════════════════════════════════════════

signal_kw = dict(arrowstyle='->', color='#2196F3', lw=1.8, zorder=4)
jam_kw = dict(arrowstyle='->', color='#E53935', lw=1.8, linestyle='dashed', zorder=4)

# ── FAS → U1 直射 ──
ax.annotate('', xy=(2.5, 1.1), xytext=(uav_x-1.0, fas_y-0.2),
            arrowprops=dict(**signal_kw, connectionstyle='arc3,rad=0.2'))
ax.text(3.8, 5.5, '$\\mathbf{h}^{U}_{U,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ── FAS → U2 直射 ──
ax.annotate('', xy=(12.5, 1.1), xytext=(uav_x+1.0, fas_y-0.2),
            arrowprops=dict(**signal_kw, connectionstyle='arc3,rad=-0.2'))
ax.text(11.2, 5.5, '$\\mathbf{h}^{U}_{U,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ── FAS → E 直射 (窃听) ──
ax.annotate('', xy=(eve_x, eve_y+0.3), xytext=(uav_x, fas_y-0.2),
            arrowprops=dict(**jam_kw))
ax.text(8.0, 5.5, '$\\mathbf{h}^{U}_{U,p}$', fontsize=9, color='#E53935',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ── FAS → RIS (上行) ──
ax.annotate('', xy=(ris_x-0.6, ris_y+ris_h/2), xytext=(uav_x-0.3, fas_y-0.2),
            arrowprops=dict(**signal_kw, connectionstyle='arc3,rad=0.05'))
ax.text(6.2, 7.0, '$\\mathbf{h}^{U}_{U,R}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ── RIS → U1 (反射) ──
ax.annotate('', xy=(2.5, 1.1), xytext=(ris_x-1.0, ris_y+ris_h/2),
            arrowprops=dict(**signal_kw, connectionstyle='arc3,rad=0.15'))
ax.text(3.5, 3.5, '$\\mathbf{h}^{U}_{R,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ── RIS → U2 (反射) ──
ax.annotate('', xy=(12.5, 1.1), xytext=(ris_x+1.0, ris_y+ris_h/2),
            arrowprops=dict(**signal_kw, connectionstyle='arc3,rad=-0.15'))
ax.text(11.5, 3.5, '$\\mathbf{h}^{U}_{R,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ── RIS → E (干扰) ──
ax.annotate('', xy=(eve_x-0.1, eve_y+0.3), xytext=(ris_x, ris_y-ris_h/2),
            arrowprops=dict(**jam_kw))
ax.text(7.2, 3.2, '$\\mathbf{h}^{U}_{R,p}$', fontsize=9, color='#E53935',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1, boxstyle='round,pad=0.2'))

# ═══════════════════════════════════════════
#  高度标注
# ═══════════════════════════════════════════
ax.annotate('', xy=(0.1, 9.2), xytext=(0.1, 0.5),
            arrowprops=dict(arrowstyle='<->', color='#999', lw=1, linestyle='--'))
ax.text(0.1, 4.8, '$H$=50m', fontsize=8, ha='center', va='center',
        color='#666', rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))

# RIS高度标注
ax.annotate('', xy=(ris_bld_x+ris_bld_w+0.6, ris_bld_y+ris_bld_h),
            xytext=(ris_bld_x+ris_bld_w+0.6, ris_bld_y),
            arrowprops=dict(arrowstyle='<->', color='#D4760A', lw=1, linestyle='--'))
ax.text(ris_bld_x+ris_bld_w+0.9, ris_bld_y+ris_bld_h/2, '12.5m', fontsize=7,
        ha='center', va='center', color='#D4760A', rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

# ═══════════════════════════════════════════
#  DRL 框图 (顶部)
# ═══════════════════════════════════════════
drl_x, drl_y = 7.5, 10.8
drl_w, drl_h = 5.0, 0.8

drl_box = FancyBboxPatch((drl_x-drl_w/2, drl_y-drl_h/2), drl_w, drl_h,
                         boxstyle="round,pad=0.08",
                         facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1.2, zorder=5)
ax.add_patch(drl_box)
ax.text(drl_x, drl_y+drl_h/2-0.12, 'Dual-Agent Twin-TD3', fontsize=10,
        ha='center', va='top', fontweight='bold', zorder=6)

a1_box = FancyBboxPatch((drl_x-2.2, drl_y-0.35), 1.6, 0.4,
                        boxstyle="round,pad=0.05",
                        facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=1, zorder=6)
ax.add_patch(a1_box)
ax.text(drl_x-1.4, drl_y-0.15, 'Agent 1\n(FAS+RIS)', fontsize=7,
        ha='center', va='center', color='#1565C0', zorder=7)

a2_box = FancyBboxPatch((drl_x+0.6, drl_y-0.35), 1.6, 0.4,
                        boxstyle="round,pad=0.05",
                        facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=1, zorder=6)
ax.add_patch(a2_box)
ax.text(drl_x+1.4, drl_y-0.15, 'Agent 2\n(UAV)', fontsize=7,
        ha='center', va='center', color='#2E7D32', zorder=7)

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

params = '$f_c$=28GHz | FAS: $N$=12 | RIS: $M$=64 | $K$=2 | $P$=1 | $H$=50m'
ax.text(7.5, 10.0, params, fontsize=7.5, ha='center', va='center', color='#666',
        bbox=dict(fc='#f5f5f5', ec='#ccc', alpha=0.9, boxstyle='round,pad=0.3', lw=0.8))

# ═══════════════════════════════════════════
#  区域标注
# ═══════════════════════════════════════════
zone1 = FancyBboxPatch((1.2, -0.2), 3.0, 1.8, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#2196F3', linewidth=0.8,
                       linestyle='--', alpha=0.4, zorder=2)
ax.add_patch(zone1)
ax.text(2.7, 1.8, '信号增强区', fontsize=7, ha='center', color='#2196F3', alpha=0.6, zorder=3)

zone2 = FancyBboxPatch((6.2, -0.2), 2.8, 1.8, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#E53935', linewidth=0.8,
                       linestyle='--', alpha=0.4, zorder=2)
ax.add_patch(zone2)
ax.text(7.6, 1.8, '干扰抑制区', fontsize=7, ha='center', color='#E53935', alpha=0.6, zorder=3)

# ═══════════════════════════════════════════
#  保存
# ═══════════════════════════════════════════
output_dir = r"C:\Users\红\Desktop\0606强化学习用于保密能源——流体天线辅助无人机保密通信"
fig.savefig(f"{output_dir}/系统模型_final.pdf", bbox_inches='tight', pad_inches=0.3, dpi=300)
fig.savefig(f"{output_dir}/系统模型_final.png", bbox_inches='tight', pad_inches=0.3, dpi=300)
print(f"PDF: {output_dir}/系统模型_final.pdf")
print(f"PNG: {output_dir}/系统模型_final.png")
plt.close()
