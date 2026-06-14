"""续训脚本：UAV_FAS (基站+流体天线) + RIS接近度奖励"""
import os, sys, csv, glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.argv = ['train_chunk.py', '--drl', 'td3', '--reward', 'see']

import numpy as np, math
from src.agents.td3_agent import Agent
from src.envs.uav_comm_env_legacy import MiniSystem

TOTAL = 2000
BASE = 'data/storage/scratch'
PROJECT = 'td3_see_fas_ris'

def count_eps(path):
    p = os.path.join(path, 'step_num_per_episode.csv')
    if not os.path.exists(p): return 0
    with open(p) as f: return sum(1 for _ in csv.reader(f))

def find_latest():
    dirs = sorted(glob.glob(os.path.join(BASE, f'{PROJECT}*')),
                  key=os.path.getmtime, reverse=True)
    for d in dirs:
        if count_eps(d) > 0 and os.path.exists(os.path.join(d, 'Actor_G_and_Phi_TD3')):
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

# Build system — UAV_FAS: BS 4天线 + FAS 4单元
system = MiniSystem(
    user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
    if_with_RIS=True, if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design='see', project_name=f'scratch/{PROJECT}', step_num=100)

if load_path:
    system.data_manager.store_path = load_path

# Agent 1: G(4×2=16实数) + F(4×2=16实数) + Phi(4) = 36维动作
# 状态维度: 2*(2+1)*(4+4)+3 = 51
agent_1 = Agent(alpha=1e-4, beta=1e-3, input_dims=[51], tau=1e-3,
    env=system, batch_size=64, layer1_size=800, layer2_size=600,
    layer3_size=512, layer4_size=256, n_actions=36, max_size=100000,
    agent_name="G_and_Phi")
# Agent 2: UAV轨迹 (2维动作)
agent_2 = Agent(alpha=1e-4, beta=1e-3, input_dims=[12], tau=1e-3,
    env=system, batch_size=64, layer1_size=400, layer2_size=300,
    layer3_size=256, layer4_size=128, n_actions=2, max_size=100000,
    agent_name="UAV")

if load_path:
    agent_1.load_models(
        load_file_actor=os.path.join(load_path, 'Actor_G_and_Phi_TD3'),
        load_file_critic_1=os.path.join(load_path, 'Critic_1_G_and_Phi_TD3'),
        load_file_critic_2=os.path.join(load_path, 'Critic_2_G_and_Phi_TD3'))
    agent_2.load_models(
        load_file_actor=os.path.join(load_path, 'Actor_UAV_TD3'),
        load_file_critic_1=os.path.join(load_path, 'Critic_1_UAV_TD3'),
        load_file_critic_2=os.path.join(load_path, 'Critic_2_UAV_TD3'))

print(f"Training from ep {existing} to {TOTAL}")

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
        # 改进：更激进地衰减探索率
        g1 = 0.05 * max(0.01, (1 - ep/800))  # 800轮后降到最低
        g2 = 0.1 * max(0.01, (1 - ep/800))
        a1 = agent_1.choose_action(o1, greedy=g1)  # 36维
        a2 = agent_2.choose_action(o2, greedy=g2)  # 2维

        # 解析36维动作: G(16) + F(16) + Phi(4)
        G_actions = a1[:16]    # BS波束成形 (4天线×2用户×2实虚=16)
        F_actions = a1[16:32]  # FAS流体天线 (4单元×2用户×2实虚=16)
        Phi_actions = a1[32:36]  # RIS相位 (4)

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
    print(f"ep_num: {ep}   ep_score: {sc}", flush=True)
    ep += 1
    if ep % 10 == 0:
        agent_1.save_models(); agent_2.save_models()

agent_1.save_models(); agent_2.save_models()
print(f"Chunk done. Total: {ep} episodes")
