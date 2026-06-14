"""ARIS训练脚本：有源RIS + UAV_FAS + 窃听者抑制
v15: RIS接近度奖励(0.5)引导UAV靠近RIS，同时保持SSR和窃听者抑制"""
import os, sys, csv, glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.argv = ['train_aris.py', '--drl', 'td3', '--reward', 'see']

import numpy as np, math
from src.agents.td3_agent import Agent
from src.envs.uav_comm_env_legacy import MiniSystem

TOTAL = 1000
BASE = 'data/storage/scratch'
PROJECT = 'td3_mfris_fas_v1'

def count_eps(path):
    p = os.path.join(path, 'step_num_per_episode.csv')
    if not os.path.exists(p): return 0
    with open(p) as f: return sum(1 for _ in csv.reader(f))

def find_latest():
    dirs = sorted(glob.glob(os.path.join(BASE, f'{PROJECT}*')),
                  key=os.path.getmtime, reverse=True)
    for d in dirs:
        if count_eps(d) > 0:
            # 检查所有必需的模型文件都存在
            required = ['Actor_G_and_Phi_TD3', 'Critic_1_G_and_Phi_TD3', 'Critic_2_G_and_Phi_TD3',
                        'Actor_UAV_TD3', 'Critic_1_UAV_TD3', 'Critic_2_UAV_TD3']
            if all(os.path.exists(os.path.join(d, f)) for f in required):
                return d
    return None

load_path = find_latest()
existing = count_eps(load_path) if load_path else 0

if load_path:
    print(f"Checkpoint: {load_path} ({existing} episodes)")
    if existing >= TOTAL:
        print("Already at target!"); sys.exit(0)
else:
    print("No existing checkpoint, starting fresh")

system = MiniSystem(
    user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
    if_with_RIS=True, if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design='ssr', project_name=f'scratch/{PROJECT}', step_num=100)

if load_path:
    system.data_manager.store_path = load_path

# Agent 1: G(16) + F(32) + RIS(8) = 56维动作, 状态77维
agent_1 = Agent(alpha=3e-4, beta=3e-3, input_dims=[77], tau=5e-4,
    env=system, batch_size=256, layer1_size=400, layer2_size=300,
    layer3_size=256, layer4_size=128, n_actions=56, max_size=500000,
    agent_name="G_and_Phi")

# Agent 2: UAV轨迹 (2维动作), 状态12维
agent_2 = Agent(alpha=3e-4, beta=3e-3, input_dims=[12], tau=5e-4,
    env=system, batch_size=256, layer1_size=256, layer2_size=128,
    layer3_size=64, layer4_size=32, n_actions=2, max_size=500000,
    agent_name="UAV")

if load_path:
    agent_1.load_models(
        load_file_actor=os.path.join(load_path, 'Actor_G_and_Phi_TD3'),
        load_file_critic_1=os.path.join(load_path, 'Critic_1_G_and_Phi_TD3'),
        load_file_critic_2=os.path.join(load_path, 'Critic_2_G_and_Phi_TD3'))
    if os.path.exists(os.path.join(load_path, 'Actor_UAV_TD3')):
        agent_2.load_models(
            load_file_actor=os.path.join(load_path, 'Actor_UAV_TD3'),
            load_file_critic_1=os.path.join(load_path, 'Critic_1_UAV_TD3'),
            load_file_critic_2=os.path.join(load_path, 'Critic_2_UAV_TD3'))
    else:
        print("Agent 2 (UAV) models not found, starting fresh for Agent 2")

print(f"Training from ep {existing} to {TOTAL}")
print("v16: Multi-functional RIS + Discrete-port FAS")

ep = existing
while ep < TOTAL:
    system.reset()
    sc = 0
    tmp = system.observe()
    z = np.random.normal(size=len(tmp))
    o1 = list(np.array(tmp) + 0.6e-7 * z)
    o2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
    for u in system.user_list: o2 += list(u.coordinate)

    for s in range(100):
        # 探索噪声: 前期充分探索，后期逐渐减小
        g1 = 0.5 * max(0.05, (1 - ep/TOTAL))
        g2 = 0.8 * max(0.05, (1 - ep/TOTAL))
        a1 = agent_1.choose_action(o1, greedy=g1)
        a2 = agent_2.choose_action(o2, greedy=g2)

        # 解析56维动作: G(16) + F(32) + RIS(8)
        G_actions = a1[:16]
        F_actions = a1[16:48]
        Phi_actions = a1[48:56]

        ns1, r, d, _ = system.step(
            action_0=a2[0], action_1=a2[1],
            G=G_actions, F=F_actions, Phi=Phi_actions,
            set_pos_x=a2[0], set_pos_y=a2[1])
        ns2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
        for u in system.user_list: ns2 += list(u.coordinate)
        sc += r
        agent_1.remember(o1, a1, r, ns1, int(d))
        agent_2.remember(o2, a2, r, ns2, int(d))
        agent_1.learn(); agent_2.learn()
        o1, o2 = ns1, ns2
        if d: break

    system.data_manager.save_file(episode_cnt=ep)
    system.reset()

    if ep % 100 == 0 and ep > 0:
        ris_dist = ((system.UAV.coordinate[0] - system.RIS.coordinate[0])**2 +
                    (system.UAV.coordinate[1] - system.RIS.coordinate[1])**2) ** 0.5
        print(f"ep: {ep}  score: {sc:.2f}  RIS_dist: {ris_dist:.1f}", flush=True)
    else:
        print(f"ep_num: {ep}   ep_score: {sc}", flush=True)

    ep += 1
    if ep % 10 == 0:
        agent_1.save_models(); agent_2.save_models()

agent_1.save_models(); agent_2.save_models()
print(f"ARIS training done. Total: {ep} episodes")
