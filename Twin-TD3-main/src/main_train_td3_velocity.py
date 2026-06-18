# TD3训练 - 速度约束模型 (方案3)
# Actor输出速度(vx,vy)，转换为位移(dx,dy)后传入环境
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import math
import torch

from src.agents.td3_agent import Agent
from src.envs.uav_comm_env import MiniSystem

parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=str, default='see', help="reward design ['ssr', 'see']")
parser.add_argument('--ep-num', type=int, default=350, help="training episodes")
parser.add_argument('--load-path', type=str, default=None, help="path to continue training")
args = parser.parse_args()

REWARD_DESIGN = args.reward
EPISODE_NUM = args.ep_num
LOAD_PATH = args.load_path

# ========== 速度约束参数 ==========
# 注意: V_MAX 仅用于信息展示，实际速度处理在环境 step() 中统一完成
# 环境中 V_MAX=1.0m/s, DT=0.1s, 单步最大位移=0.1m
V_MAX = 1.0       # 与环境保持一致 (m/s)
DT = 0.1          # 与环境保持一致 (s)
STEP_SCALE = V_MAX * DT  # 单步最大位移 = 0.1m
MAX_DIST_PER_EP = V_MAX * DT * 100  # 每轮最大距离 = 10m

print(f"速度约束: v_max={V_MAX}m/s, dt={DT}s, 单步最大={STEP_SCALE}m, 每轮最大={MAX_DIST_PER_EP}m")

# ========== 初始化系统 ==========
system = MiniSystem(
    user_num=2, ris_ant_num=64,
    if_dir_link=1, if_with_FAS=True,
    if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design=REWARD_DESIGN, project_name=None, step_num=100,
    total_episodes=EPISODE_NUM
)

# ========== Agent 1: FAS波束成形 ==========
agent_1 = Agent(
    alpha=1e-4, beta=1e-3,
    input_dims=[system.get_system_state_dim()],
    tau=0.005, env=system, batch_size=256,
    layer1_size=1024, layer2_size=768,
    layer3_size=512, layer4_size=256,
    n_actions=system.get_system_action_dim(),  # 12端口 + 1 F + 1 β + 1 η + 24 RIS + 4用户权重 = 43
    max_size=1000000,
    agent_name="FAS"
)

# ========== Agent 2: UAV速度控制 ==========
# 输出2维速度(vx,vy)，不是位移(dx,dy)
agent_2 = Agent(
    alpha=1e-4, beta=1e-3,
    input_dims=[system.get_uav_local_state_dim()],
    tau=0.005, env=system, batch_size=256,
    layer1_size=512, layer2_size=384,
    layer3_size=256, layer4_size=128,
    n_actions=2,  # vx, vy (速度，不是位移)
    max_size=1000000,
    agent_name="UAV"
)

# ========== 加载模型 ==========
existing_episode_count = 0
if LOAD_PATH and os.path.exists(LOAD_PATH):
    import glob
    mat_files = glob.glob(os.path.join(LOAD_PATH, 'simulation_result_ep_*.mat'))
    existing_episode_count = len(mat_files)
    print(f"Resuming from episode {existing_episode_count}...")

    agent_1.load_models(
        load_file_actor=os.path.join(LOAD_PATH, 'Actor_BS_FAS_TD3'),
        load_file_critic_1=os.path.join(LOAD_PATH, 'Critic_1_BS_FAS_TD3'),
        load_file_critic_2=os.path.join(LOAD_PATH, 'Critic_2_BS_FAS_TD3')
    )
    agent_2.load_models(
        load_file_actor=os.path.join(LOAD_PATH, 'Actor_UAV_TD3'),
        load_file_critic_1=os.path.join(LOAD_PATH, 'Critic_1_UAV_TD3'),
        load_file_critic_2=os.path.join(LOAD_PATH, 'Critic_2_UAV_TD3')
    )

# ========== 打印信息 ==========
print("=" * 60)
print("TD3 Velocity Constraint Training - FAS-UAV Secure Communication")
print("=" * 60)
print(f"State dim (Agent 1): {system.get_system_state_dim()}")
print(f"State dim (Agent 2): {system.get_uav_local_state_dim()}")
print(f"Action dim (Agent 1): {system.get_system_action_dim()} (端口+F+β+η+RIS相位+用户权重)")
print(f"Action dim (Agent 2): 2 (归一化速度) → 环境统一处理位移")
print(f"Speed limit: v_max={V_MAX}m/s, dt={DT}s")
print(f"Step scale: {STEP_SCALE}m (单步最大位移)")
print(f"Max distance per episode: {MAX_DIST_PER_EP}m")
print(f"Episodes: {EPISODE_NUM}")
print("=" * 60)

# ========== 训练循环 ==========
episode_cnt = existing_episode_count
step_num = 100

while episode_cnt < EPISODE_NUM:
    system.reset()
    step_cnt = 0
    score_per_ep = 0
    total_distance = 0  # 记录本轮总移动距离

    # 获取初始观测
    obs1 = system.observe()
    obs2 = system.observe_uav_local()

    while step_cnt < step_num:
        step_cnt += 1

        if not system.render_obj.pause:
            # Agent 1: 波束成形 (直接输出)
            a1 = agent_1.choose_action(obs1, greedy=0.1 * math.pow(1 - episode_cnt / EPISODE_NUM, 2))

            # Agent 2: 速度控制
            # 直接输出归一化速度 [-1, 1]，由环境统一处理速度→位移转换
            a2_raw = agent_2.choose_action(obs2, greedy=0.5 * math.pow(1 - episode_cnt / EPISODE_NUM, 2))

            # 传入环境 (环境内部统一处理: action → 速度 → 位移 → 边界约束)
            action_0 = np.clip(a2_raw[0], -1, 1)
            action_1 = np.clip(a2_raw[1], -1, 1)

            # 执行动作
            # 动作维度: 43维 = G[0:13](端口12+F增益1) + Phi[13:39](β+η+RIS相位26) + 用户权重[39:43](K×num_active_ports)
            new_s1, reward, done, _ = system.step(
                action_0=action_0, action_1=action_1, action_2=0,
                G=list(a1[:13]), Phi=list(a1[13:39]), user_weights=list(a1[39:43])
            )
            new_s2 = system.observe_uav_local()

            # 记录移动距离 (从环境获取实际位移)
            move_data = system.data_manager.simulation_result_dic.get('UAV_movement', [])
            if len(move_data) > 0:
                last_move = move_data[-1]
                step_dist = math.sqrt(last_move[0]**2 + last_move[1]**2)
            else:
                step_dist = 0
            total_distance += step_dist

            score_per_ep += reward

            # 存储经验 (存储原始速度动作，不是位移)
            agent_1.remember(obs1, a1, reward, new_s1, int(done))
            agent_2.remember(obs2, a2_raw, reward, new_s2, int(done))

            # 学习: Agent 1正常学，Agent 2自适应频率（前期少学，后期正常学）
            agent_1.learn()
            learn_ratio = min(1.0, episode_cnt / (EPISODE_NUM * 0.3))  # 前30%逐渐增加
            if np.random.random() < learn_ratio:
                agent_2.learn()

            obs1 = new_s1
            obs2 = new_s2

            if done:
                break
        else:
            import time
            time.sleep(0.001)

    # 保存数据
    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()

    print(f"ep: {episode_cnt:4d} | reward: {score_per_ep:8.3f} | dist: {total_distance:.1f}m", flush=True)
    episode_cnt += 1

    # 定期保存模型
    if episode_cnt % 10 == 0:
        agent_1.save_models()
        agent_2.save_models()

# 保存最终模型
agent_1.save_models()
agent_2.save_models()
print(f"\nTraining complete! {episode_cnt} episodes")
