#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 UAV-FAS 一体化系统
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from src.envs.uav_comm_env import MiniSystem
import numpy as np

print("="*60)
print("测试 UAV-FAS 一体化系统")
print("="*60)

# 创建系统实例
system = MiniSystem(
    user_num=2,
    fas_ant_num=12,
    if_dir_link=1,
    if_with_FAS=True,
    if_move_users=True,
    if_movements=True,
    reverse_x_y=(False, False),
    if_UAV_pos_state=True,
    reward_design='see',
    project_name='test_uav_fas',
    step_num=10
)

print("\n[系统初始化信息]")
print(f"- 用户数量: {system.user_num}")
print(f"- 攻击者数量: {system.attacker_num}")
print(f"- FAS端口数: {system.UAV_FAS.fas_num_ports}")
print(f"- 是否启用FAS: {system.if_with_FAS}")
print(f"- 动作维度: {system.get_system_action_dim()}")
print(f"- 状态维度: {system.get_system_state_dim()}")

# 测试重置功能
print("\n[测试系统重置]")
system.reset()
print("[OK] 系统重置成功")

# 测试状态观测
print("\n[测试状态观测]")
state = system.observe()
print(f"[OK] 状态观测成功，维度: {len(state)}")

# 测试单步执行
print("\n[测试单步执行]")
fas_action_dim = system.UAV_FAS.fas_num_ports + 1  # 12端口选择 + 1 F增益 = 13
ris_action_dim = 25  # 1 beta + 24 phases
action_1 = np.zeros(fas_action_dim + ris_action_dim)  # 38维
action_2 = np.array([0.1, 0.1])  # UAV移动动作

new_state, reward, done, info = system.step(
    action_0=action_2[0],
    action_1=action_2[1],
    G=action_1[:fas_action_dim],
    Phi=action_1[fas_action_dim:]
)

print(f"[OK] 单步执行成功")
print(f"  - 奖励值: {reward:.4f}")
print(f"  - 是否终止: {done}")
print(f"  - 新状态维度: {len(new_state)}")

# 测试多步执行
print("\n[测试多步执行]")
for i in range(5):
    action_1 = np.random.uniform(-1, 1, fas_action_dim + ris_action_dim)
    action_2 = np.random.uniform(-1, 1, 2)

    new_state, reward, done, info = system.step(
        action_0=action_2[0],
        action_1=action_2[1],
        G=action_1[:fas_action_dim],
        Phi=action_1[fas_action_dim:]
    )

    print(f"  步骤 {i+1}: 奖励={reward:.4f}, 终止={done}")

print("\n" + "="*60)
print("[OK] 所有测试通过！UAV-FAS 系统运行正常")
print("="*60)