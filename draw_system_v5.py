#!/usr/bin/env python3
"""
系统模型图 v5 — 修正版
1. 标题不被DRL框挡住, 图例排版整齐
2. 用户改为 User 0, User 1, 旁边无建筑
3. RIS贴合在25m建筑的12.5m高处
4. 窃听者在不同3D位置, 不与建筑贴合
5. 无人机3D效果, 文字与机身契合
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Ellipse
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 11), dpi=250)
ax.set_xlim(-1.0, 16.0)
ax.set_ylim(-2.0, 12.5)
ax.set_aspect('equal')
ax.axis('off')

# ═══════════════════════════════════════════
#  背景
# ═══════════════════════════════════════════
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack([gradient]*10)
ax.imshow(gradient, aspect='auto', cmap=plt.cm.Blues,
          extent=[-1.0, 16.0, 10.5, 12.5], alpha=0.15, zorder=0)

# 地面
ground = FancyBboxPatch((-1.0, -2.0), 17, 2.5, boxstyle="square,pad=0",
                        facecolor='#D7CCC8', edgecolor='#8D6E63', linewidth=1.5, zorder=1)
ax.add_patch(ground)
for gx in np.arange(-0.8, 16.0, 0.6):
    ax.plot([gx, gx+0.25], [-0.1, -0.6], color='#BCAAA4', lw=0.4, alpha=0.5, zorder=1)

# ═══════════════════════════════════════════
#  建筑物 (只有中间一栋, 25m高 = 约8个单位)
# ═══════════════════════════════════════════
bld_x, bld_y = 5.8, 0.0
bld_w, bld_h = 3.4, 8.0  # 25m高

# 正面
bld_front = FancyBboxPatch((bld_x, bld_y), bld_w, bld_h,
                           boxstyle="square,pad=0",
                           facecolor='#B0BEC5', edgecolor='#78909C', linewidth=1.2, zorder=2)
ax.add_patch(bld_front)
# 3D侧面 (右侧)
side_x = [bld_x+bld_w, bld_x+bld_w+0.5, bld_x+bld_w+0.5, bld_x+bld_w]
side_y = [bld_y, bld_y+0.35, bld_y+bld_h+0.35, bld_y+bld_h]
side = Polygon(list(zip(side_x, side_y)), closed=True,
               facecolor='#90A4AE', edgecolor='#78909C', linewidth=0.8, zorder=2)
ax.add_patch(side)
# 顶面
top_x = [bld_x, bld_x+0.5, bld_x+bld_w+0.5, bld_x+bld_w]
top_y = [bld_y+bld_h, bld_y+bld_h+0.35, bld_y+bld_h+0.35, bld_y+bld_h]
top_poly = Polygon(list(zip(top_x, top_y)), closed=True,
                   facecolor='#CFD8DC', edgecolor='#78909C', linewidth=0.8, zorder=2)
ax.add_patch(top_poly)
# 窗户
for row in range(9):
    for col in range(4):
        wx = bld_x + 0.2 + col * 0.8
        wy = bld_y + 0.2 + row * 0.88
        if wy + 0.6 < bld_y + bld_h:
            win = Rectangle((wx, wy), 0.5, 0.6, facecolor='#B3E5FC',
                           edgecolor='#90A4AE', linewidth=0.3, alpha=0.7, zorder=3)
            ax.add_patch(win)

# 25m高度标注
ax.annotate('', xy=(bld_x+bld_w+0.8, bld_y+bld_h),
            xytext=(bld_x+bld_w+0.8, bld_y),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax.text(bld_x+bld_w+1.1, bld_y+bld_h/2, '25m', fontsize=8, ha='center', va='center',
        color='#666', rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

# ═══════════════════════════════════════════
#  主动 RIS (贴合在建筑12.5m高处 = 约4个单位高)
# ═══════════════════════════════════════════
ris_x = bld_x + bld_w / 2  # 建筑中心
ris_y = bld_y + 4.0  # 12.5m ≈ 4个单位 (建筑总高8单位=25m)
ris_w, ris_h = 2.8, 0.55

# RIS 面板 (紧贴建筑正面)
ris_panel = FancyBboxPatch((ris_x-ris_w/2, ris_y-ris_h/2), ris_w, ris_h,
                           boxstyle="round,pad=0.03",
                           facecolor='#D4760A', edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(ris_panel)
# RIS 单元格
for row in range(2):
    for col in range(8):
        cx = ris_x - ris_w/2 + 0.1 + col * (ris_w-0.2)/8
        cy = ris_y - ris_h/2 + 0.06 + row * (ris_h-0.12)/2
        cell = Rectangle((cx, cy), (ris_w-0.25)/8-0.02, (ris_h-0.15)/2-0.02,
                         facecolor='#FFB74D', edgecolor='#D4760A',
                         linewidth=0.3, alpha=0.8, zorder=6)
        ax.add_patch(cell)

ax.text(ris_x, ris_y+ris_h/2+0.15, '主动 RIS ($M$=64 单元)', fontsize=9,
        ha='center', va='bottom', color='#D4760A', fontweight='bold', zorder=6)
ax.text(ris_x, ris_y-ris_h/2-0.08, '信号反射 + 人工噪声', fontsize=7,
        ha='center', va='top', color='#BF360C', zorder=6)

# 12.5m高度标注 (RIS位置)
ax.annotate('', xy=(bld_x+bld_w+0.8, ris_y),
            xytext=(bld_x+bld_w+0.8, bld_y),
            arrowprops=dict(arrowstyle='<->', color='#D4760A', lw=1, linestyle='--'))
ax.text(bld_x+bld_w+1.1, ris_y/2, '12.5m', fontsize=7, ha='center', va='center',
        color='#D4760A', rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

# ═══════════════════════════════════════════
#  无人机 (3D效果)
# ═══════════════════════════════════════════
uav_x, uav_y = 7.5, 10.0

# 阴影 (更明显)
shadow = Ellipse((uav_x+0.2, uav_y-0.2), 1.6, 0.6, angle=-10,
                 facecolor='black', alpha=0.1, zorder=3)
ax.add_patch(shadow)

# 旋翼 (3D椭圆, 带运动模糊)
rotor_positions = [(-1.2, 0.6, 15), (1.2, 0.6, -15), (-1.2, -0.6, -15), (1.2, -0.6, 15)]
for rx, ry, rot in rotor_positions:
    # 旋翼外圈 (模糊效果)
    rotor_outer = Ellipse((uav_x+rx, uav_y+ry), 0.9, 0.25, angle=rot,
                          facecolor='#90CAF9', edgecolor='none', alpha=0.3, zorder=3)
    ax.add_patch(rotor_outer)
    # 旋翼本体
    rotor = Ellipse((uav_x+rx, uav_y+ry), 0.75, 0.2, angle=rot,
                    facecolor='#5C8BBF', edgecolor='#2C5AA0', linewidth=0.8, alpha=0.7, zorder=4)
    ax.add_patch(rotor)
    # 旋翼臂
    ax.plot([uav_x+rx*0.45, uav_x+rx*0.8], [uav_y+ry*0.45, uav_y+ry*0.8],
            color='#2C5AA0', lw=2, alpha=0.6, zorder=4)

# 机身 (3D效果: 主体+侧面+底面)
# 主体
body_main = FancyBboxPatch((uav_x-0.55, uav_y-0.22), 1.1, 0.44,
                           boxstyle="round,pad=0.06",
                           facecolor='#2C5AA0', edgecolor='#1A3A6A', linewidth=1.5, zorder=5)
ax.add_patch(body_main)
# 3D侧面 (右侧)
body_side = Polygon([(uav_x+0.55, uav_y-0.22), (uav_x+0.7, uav_y-0.15),
                     (uav_x+0.7, uav_y+0.15), (uav_x+0.55, uav_y+0.22)],
                    closed=True, facecolor='#1A3A6A', edgecolor='#0D2340',
                    linewidth=0.8, zorder=5)
ax.add_patch(body_side)
# 3D底面
body_bottom = Polygon([(uav_x-0.55, uav_y-0.22), (uav_x+0.55, uav_y-0.22),
                        (uav_x+0.7, uav_y-0.15), (uav_x-0.4, uav_y-0.15)],
                      closed=True, facecolor='#1A3A6A', edgecolor='#0D2340',
                      linewidth=0.8, zorder=5)
ax.add_patch(body_bottom)

# UAV 文字 (居中在机身上)
ax.text(uav_x, uav_y, 'UAV', fontsize=12, ha='center', va='center',
        color='white', fontweight='bold', zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground='#0D2340')])

# ═══════════════════════════════════════════
#  FAS 天线阵列
# ═══════════════════════════════════════════
fas_y = uav_y - 0.55
fas_w = 2.8
fas_h = 0.22

fas_base = FancyBboxPatch((uav_x-fas_w/2, fas_y-fas_h/2), fas_w, fas_h,
                          boxstyle="round,pad=0.03",
                          facecolor='#1A8A5C', edgecolor='white', linewidth=1, zorder=5)
ax.add_patch(fas_base)
ax.text(uav_x, fas_y+fas_h/2+0.12, 'FAS ($N$=12端口)', fontsize=8, ha='center',
        va='bottom', color='#1A8A5C', fontweight='bold', zorder=6)

# 端口
n_ports = 12
port_spacing = fas_w / (n_ports + 1)
active_ports = [3, 7, 10]
for i in range(n_ports):
    px = uav_x - fas_w/2 + (i+1) * port_spacing
    active = (i+1) in active_ports
    color = '#4CAF50' if active else '#90A4AE'
    size = 0.09 if active else 0.065
    circle = plt.Circle((px, fas_y), size, color=color, ec='white',
                        lw=0.7 if not active else 1.0, zorder=6)
    ax.add_patch(circle)

# ═══════════════════════════════════════════
#  合法用户 (地面, 简单手机, User 0 和 User 1)
# ═══════════════════════════════════════════

def draw_phone(ax, x, y, label, color='#2196F3'):
    """简单手机形状"""
    # 阴影
    phone_shadow = FancyBboxPatch((x-0.13+0.04, y-0.22+0.04), 0.26, 0.44,
                                  boxstyle="round,pad=0.02",
                                  facecolor='black', alpha=0.12, zorder=3)
    ax.add_patch(phone_shadow)
    # 手机外框
    phone = FancyBboxPatch((x-0.13, y-0.22), 0.26, 0.44,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='white', linewidth=1, zorder=5)
    ax.add_patch(phone)
    # 屏幕
    screen = Rectangle((x-0.09, y-0.15), 0.18, 0.30,
                       facecolor='white', edgecolor=color, linewidth=0.4, alpha=0.9, zorder=6)
    ax.add_patch(screen)
    # 信号图标
    ax.plot([x, x], [y+0.06, y+0.11], color=color, lw=1.2, zorder=7)
    ax.plot([x-0.03, x+0.03], [y+0.11, y+0.11], color=color, lw=1, zorder=7)
    ax.plot([x-0.05, x+0.05], [y+0.14, y+0.14], color=color, lw=0.8, zorder=7)
    # 标签
    ax.text(x, y-0.32, label, fontsize=8, ha='center', va='top',
            color='#1565C0', fontweight='bold', zorder=6)

# User 0 (左侧, 远离建筑)
draw_phone(ax, 2.0, 0.8, 'User 0')

# User 1 (右侧, 远离建筑)
draw_phone(ax, 13.0, 0.8, 'User 1')

# ═══════════════════════════════════════════
#  窃听者 (地面, 不同3D位置, 远离建筑)
# ═══════════════════════════════════════════
eve_x, eve_y = 10.5, 0.8

# 阴影
eve_shadow = FancyBboxPatch((eve_x-0.13+0.04, eve_y-0.22+0.04), 0.26, 0.44,
                            boxstyle="round,pad=0.02",
                            facecolor='black', alpha=0.12, zorder=3)
ax.add_patch(eve_shadow)
# 手机外框 (红色)
eve_phone = FancyBboxPatch((eve_x-0.13, eve_y-0.22), 0.26, 0.44,
                           boxstyle="round,pad=0.02",
                           facecolor='#E53935', edgecolor='white', linewidth=1, zorder=5)
ax.add_patch(eve_phone)
# 屏幕
eve_screen = Rectangle((eve_x-0.09, eve_y-0.15), 0.18, 0.30,
                       facecolor='white', edgecolor='#E53935', linewidth=0.4, alpha=0.9, zorder=6)
ax.add_patch(eve_screen)
# 窃听图标
ax.plot([eve_x-0.04, eve_x+0.04], [eve_y+0.04, eve_y+0.04], color='#E53935', lw=1.5, zorder=7)
ax.plot([eve_x-0.02, eve_x+0.02], [eve_y+0.04, eve_y+0.04], color='#E53935', lw=3, zorder=7)
ax.text(eve_x, eve_y-0.32, 'Eavesdropper $p$', fontsize=8, ha='center', va='top',
        color='#B71C1C', fontweight='bold', zorder=6)

# ═══════════════════════════════════════════
#  信号路径
# ═══════════════════════════════════════════
sig_kw = dict(arrowstyle='->', color='#2196F3', lw=1.6, zorder=4)
jam_kw = dict(arrowstyle='->', color='#E53935', lw=1.6, linestyle='dashed', zorder=4)

# FAS → User 0 直射
ax.annotate('', xy=(2.0, 1.05), xytext=(uav_x-1.0, fas_y-0.15),
            arrowprops=dict(**sig_kw, connectionstyle='arc3,rad=0.18'))
ax.text(3.5, 6.0, '$\\mathbf{h}^{U}_{U,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# FAS → User 1 直射
ax.annotate('', xy=(13.0, 1.05), xytext=(uav_x+1.0, fas_y-0.15),
            arrowprops=dict(**sig_kw, connectionstyle='arc3,rad=-0.18'))
ax.text(11.5, 6.0, '$\\mathbf{h}^{U}_{U,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# FAS → Eve 直射 (窃听)
ax.annotate('', xy=(eve_x, eve_y+0.3), xytext=(uav_x+0.5, fas_y-0.15),
            arrowprops=dict(**jam_kw))
ax.text(9.8, 6.0, '$\\mathbf{h}^{U}_{U,p}$', fontsize=9, color='#E53935',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# FAS → RIS (上行)
ax.annotate('', xy=(ris_x-0.8, ris_y+ris_h/2), xytext=(uav_x-0.3, fas_y-0.15),
            arrowprops=dict(**sig_kw, connectionstyle='arc3,rad=0.05'))
ax.text(6.0, 8.2, '$\\mathbf{h}^{U}_{U,R}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# RIS → User 0 (反射)
ax.annotate('', xy=(2.0, 1.05), xytext=(ris_x-1.3, ris_y+ris_h/2),
            arrowprops=dict(**sig_kw, connectionstyle='arc3,rad=0.12'))
ax.text(3.0, 3.5, '$\\mathbf{h}^{U}_{R,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# RIS → User 1 (反射)
ax.annotate('', xy=(13.0, 1.05), xytext=(ris_x+1.3, ris_y+ris_h/2),
            arrowprops=dict(**sig_kw, connectionstyle='arc3,rad=-0.12'))
ax.text(12.0, 3.5, '$\\mathbf{h}^{U}_{R,k}$', fontsize=9, color='#2196F3',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# RIS → Eve (干扰)
ax.annotate('', xy=(eve_x-0.1, eve_y+0.3), xytext=(ris_x+0.5, ris_y-ris_h/2),
            arrowprops=dict(**jam_kw))
ax.text(9.5, 3.0, '$\\mathbf{h}^{U}_{R,p}$', fontsize=9, color='#E53935',
        ha='center', style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5, boxstyle='round,pad=0.2'))

# ═══════════════════════════════════════════
#  H=50m 高度标注 (左侧)
# ═══════════════════════════════════════════
ax.annotate('', xy=(-0.5, 10.0), xytext=(-0.5, 0.5),
            arrowprops=dict(arrowstyle='<->', color='#999', lw=1, linestyle='--'))
ax.text(-0.5, 5.2, '$H$=50m', fontsize=8, ha='center', va='center',
        color='#666', rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))

# ═══════════════════════════════════════════
#  DRL 框图 (顶部, 不挡住标题)
# ═══════════════════════════════════════════
drl_x, drl_y = 7.5, 11.8
drl_w, drl_h = 4.5, 0.6

drl_box = FancyBboxPatch((drl_x-drl_w/2, drl_y-drl_h/2), drl_w, drl_h,
                         boxstyle="round,pad=0.06",
                         facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1, zorder=5)
ax.add_patch(drl_box)
ax.text(drl_x, drl_y+0.12, 'Dual-Agent Twin-TD3', fontsize=9,
        ha='center', va='center', fontweight='bold', zorder=6)

a1_box = FancyBboxPatch((drl_x-2.0, drl_y-0.28), 1.4, 0.32,
                        boxstyle="round,pad=0.04",
                        facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=0.8, zorder=6)
ax.add_patch(a1_box)
ax.text(drl_x-1.3, drl_y-0.12, 'Agent 1 (FAS+RIS)', fontsize=6.5,
        ha='center', va='center', color='#1565C0', zorder=7)

a2_box = FancyBboxPatch((drl_x+0.6, drl_y-0.28), 1.4, 0.32,
                        boxstyle="round,pad=0.04",
                        facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=0.8, zorder=6)
ax.add_patch(a2_box)
ax.text(drl_x+1.3, drl_y-0.12, 'Agent 2 (UAV)', fontsize=6.5,
        ha='center', va='center', color='#2E7D32', zorder=7)

# ═══════════════════════════════════════════
#  标题 (DRL框下方)
# ═══════════════════════════════════════════
ax.text(7.5, 11.0, 'FAS辅助主动RIS无人机安全通信系统模型',
        fontsize=14, ha='center', va='center', fontweight='bold', color='#1a1a1a')

params = '$f_c$=28GHz | FAS: $N$=12 | RIS: $M$=64 | $K$=2 | $P$=1 | $H$=50m'
ax.text(7.5, 10.5, params, fontsize=7.5, ha='center', va='center', color='#666',
        bbox=dict(fc='#f5f5f5', ec='#ccc', alpha=0.9, boxstyle='round,pad=0.3', lw=0.8))

# ═══════════════════════════════════════════
#  图例 (右下角, 排版整齐)
# ═══════════════════════════════════════════
legend_x, legend_y = 13.5, 2.0
legend_items = [
    ('#2196F3', '-', '信号链路 (LOS)'),
    ('#E53935', '--', '窃听/干扰链路'),
    ('#4CAF50', 'o', 'FAS 激活端口'),
    ('#90A4AE', 'o', 'FAS 未激活端口'),
]
# 图例背景
leg_bg = FancyBboxPatch((legend_x-0.3, legend_y-len(legend_items)*0.35-0.15), 2.5, len(legend_items)*0.35+0.4,
                        boxstyle="round,pad=0.08",
                        facecolor='white', edgecolor='#BDBDBD', linewidth=0.8, alpha=0.95, zorder=5)
ax.add_patch(leg_bg)
ax.text(legend_x+0.95, legend_y+0.1, '图例', fontsize=8, ha='center', va='center',
        fontweight='bold', color='#333', zorder=6)

for i, (color, ls, label) in enumerate(legend_items):
    ly = legend_y - i * 0.35 - 0.2
    if ls == 'o':
        ax.plot(legend_x, ly, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.8, zorder=6)
    elif ls == '-':
        ax.plot([legend_x-0.15, legend_x+0.15], [ly, ly], color=color, lw=1.5, zorder=6)
    else:
        ax.plot([legend_x-0.15, legend_x+0.15], [ly, ly], color=color, lw=1.5, linestyle='--', zorder=6)
    ax.text(legend_x+0.25, ly, label, fontsize=7, va='center', color='#333', zorder=6)

# ═══════════════════════════════════════════
#  区域标注
# ═══════════════════════════════════════════
zone1 = FancyBboxPatch((0.8, -0.3), 2.8, 1.8, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#2196F3', linewidth=0.8,
                       linestyle='--', alpha=0.4, zorder=2)
ax.add_patch(zone1)
ax.text(2.2, 1.7, '信号增强区', fontsize=7, ha='center', color='#2196F3', alpha=0.6, zorder=3)

zone2 = FancyBboxPatch((9.2, -0.3), 3.0, 1.8, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#E53935', linewidth=0.8,
                       linestyle='--', alpha=0.4, zorder=2)
ax.add_patch(zone2)
ax.text(10.7, 1.7, '干扰抑制区', fontsize=7, ha='center', color='#E53935', alpha=0.6, zorder=3)

# ═══════════════════════════════════════════
#  保存
# ═══════════════════════════════════════════
output_dir = r"C:\Users\红\Desktop\0606强化学习用于保密能源——流体天线辅助无人机保密通信"
fig.savefig(f"{output_dir}/系统模型_v5.pdf", bbox_inches='tight', pad_inches=0.3, dpi=300)
fig.savefig(f"{output_dir}/系统模型_v5.png", bbox_inches='tight', pad_inches=0.3, dpi=300)
print(f"PDF: {output_dir}/系统模型_v5.pdf")
print(f"PNG: {output_dir}/系统模型_v5.png")
plt.close()
