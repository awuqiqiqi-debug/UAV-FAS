# SAC训练入口 - 基于现有系统模型
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import math
import torch

from src.agents.sac_agent import SACAgent
from src.envs.uav_comm_env import MiniSystem

parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=str, default='see', help="reward design ['ssr', 'see']")
parser.add_argument('--ep-num', type=int, default=350, help="training episodes")
parser.add_argument('--load-path', type=str, default=None, help="path to continue training")
args = parser.parse_args()

REWARD_DESIGN = args.reward
EPISODE_NUM = args.ep_num
LOAD_PATH = args.load_path

# ========== 初始化系统 ==========
system = MiniSystem(
    user_num=2, ris_ant_num=64,
    if_dir_link=1, if_with_FAS=True,
    if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design=REWARD_DESIGN, project_name=None, step_num=100
)

# ========== Agent 1: FAS波束成形 ==========
agent_1 = SACAgent(
    alpha=3e-4, beta=3e-4,  # SAC标准学习率
    input_dims=[system.get_system_state_dim()],
    tau=0.005, env=system, batch_size=256,
    layer1_size=1024, layer2_size=768,
    layer3_size=512, layer4_size=256,
    n_actions=system.get_system_action_dim(),  # 12端口 + 1 F + 1 β + 24 RIS = 38
    max_size=1000000,
    agent_name="FAS"
)

# ========== Agent 2: UAV轨迹 ==========
agent_2 = SACAgent(
    alpha=3e-4, beta=3e-4,  # SAC标准学习率
    input_dims=[system.get_uav_local_state_dim()],
    tau=0.005, env=system, batch_size=256,
    layer1_size=512, layer2_size=384,
    layer3_size=256, layer4_size=128,
    n_actions=3,  # dx, dy, dz
    max_size=1000000,
    agent_name="UAV"
)

# ========== SAC专用奖励归一化 ==========
class SACRewardNormalizer:
    """
    SAC奖励归一化策略:
    1. 将奖励偏移到正值范围
    2. 使用running statistics归一化
    3. 裁剪到合理范围
    """
    def __init__(self):
        self.reward_sum = 0
        self.reward_sq_sum = 0
        self.count = 0
        self.reward_min = float('inf')
        self.reward_max = float('-inf')

    def normalize(self, reward):
        # 更新统计量
        self.reward_sum += reward
        self.reward_sq_sum += reward ** 2
        self.count += 1
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)

        # 计算均值和标准差
        mean = self.reward_sum / self.count
        var = self.reward_sq_sum / self.count - mean ** 2
        std = max(var ** 0.5, 1e-8)

        # 归一化到 [0, 1]
        normalized = (reward - self.reward_min) / (self.reward_max - self.reward_min + 1e-8)

        # 偏移到正值范围 (SAC需要正奖励)
        normalized = normalized + 0.5  # 偏移到 [0.5, 1.5]

        return normalized

sac_reward_normalizer = SACRewardNormalizer()

# ========== 加载模型 ==========
existing_episode_count = 0
if LOAD_PATH and os.path.exists(LOAD_PATH):
    import glob
    mat_files = glob.glob(os.path.join(LOAD_PATH, 'simulation_result_ep_*.mat'))
    existing_episode_count = len(mat_files)
    print(f"Resuming from episode {existing_episode_count}...")

    agent_1.load_models(
        load_file_actor=os.path.join(LOAD_PATH, 'Actor_BS_FAS_SAC'),
        load_file_critic_1=os.path.join(LOAD_PATH, 'Critic_1_BS_FAS_SAC'),
        load_file_critic_2=os.path.join(LOAD_PATH, 'Critic_2_BS_FAS_SAC')
    )
    agent_2.load_models(
        load_file_actor=os.path.join(LOAD_PATH, 'Actor_UAV_SAC'),
        load_file_critic_1=os.path.join(LOAD_PATH, 'Critic_1_UAV_SAC'),
        load_file_critic_2=os.path.join(LOAD_PATH, 'Critic_2_UAV_SAC')
    )

# ========== 打印信息 ==========
print("=" * 60)
print("SAC Training - FAS-UAV Secure Communication")
print("=" * 60)
print(f"State dim (Agent 1): {system.get_system_state_dim()}")
print(f"State dim (Agent 2): {system.get_uav_local_state_dim()}")
print(f"Action dim (Agent 1): 48 (BS+FAS+RIS)")
print(f"Action dim (Agent 2): 3 (dx, dy, dz)")
print(f"Episodes: {EPISODE_NUM}")
print(f"Border: X={system.border[0]}, Y={system.border[1]}")
print("=" * 60)

# ========== 训练循环 ==========
episode_cnt = existing_episode_count
step_num = 100

while episode_cnt < EPISODE_NUM:
    system.reset()
    step_cnt = 0
    score_per_ep = 0

    # 获取初始观测
    obs1 = system.observe()
    obs2 = system.observe_uav_local()

    while step_cnt < step_num:
        step_cnt += 1

        if not system.render_obj.pause:
            # 选择动作（SAC天然随机探索，无需额外噪声）
            a1 = agent_1.choose_action(obs1)
            a2 = agent_2.choose_action(obs2)

            # 执行动作
            new_s1, reward, done, _ = system.step(
                action_0=a2[0], action_1=a2[1], action_2=a2[2],
                G=list(a1[:13]), Phi=list(a1[13:38])
            )
            new_s2 = system.observe_uav_local()

            # SAC专用奖励归一化
            norm_reward = sac_reward_normalizer.normalize(reward)
            score_per_ep += reward  # 记录原始奖励

            # 存储经验 (使用归一化后奖励)
            agent_1.remember(obs1, a1, norm_reward, new_s1, int(done))
            agent_2.remember(obs2, a2, norm_reward, new_s2, int(done))

            # 学习
            agent_1.learn()
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

    print(f"ep: {episode_cnt:4d} | reward: {score_per_ep:8.3f}", flush=True)
    episode_cnt += 1

    # 定期保存模型
    if episode_cnt % 10 == 0:
        agent_1.save_models()
        agent_2.save_models()

# 保存最终模型
agent_1.save_models()
agent_2.save_models()
print(f"\nTraining complete! {episode_cnt} episodes")
