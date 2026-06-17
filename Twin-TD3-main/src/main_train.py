# debug field
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drl', type = str, required = True, default='td3', help="which drl algo would you like to choose ['ddpg', 'td3']")
parser.add_argument('--reward', type = str, required = True, default='see', help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--seeds', type = int, required = False, default=None,  nargs='+', help="what seed(s) would you like to use for DRL 1 and 2")
parser.add_argument('--ep-num', type = int, required = False, default=2000, help="how many episodes do you want to train your DRL")
parser.add_argument('--trained-uav', default=False, action='store_true', help='use trained uav instead of retraining')
parser.add_argument('--load-path', type = str, required = False, default=None, help="path to existing training directory to continue training")

args = parser.parse_args()
DRL_ALGO = args.drl
REWARD_DESIGN = args.reward
SEEDS = args.seeds
EPISODE_NUM = args.ep_num
TRAINED_UAV = args.trained_uav
LOAD_PATH = args.load_path

assert DRL_ALGO in ['ddpg', 'td3'], "drl must be ['ddpg', 'td3']"
assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"
if SEEDS is not None:
    assert len(SEEDS) in [1, 2] and isinstance(SEEDS[0], int) and isinstance(SEEDS[-1], int), "seeds must be a list of 1 or 2 integer"

if DRL_ALGO == 'td3':
    from src.agents.td3_agent import Agent
elif DRL_ALGO == 'ddpg':
    from src.agents.ddpg_agent import Agent

# 导入 UAV-FAS 环境
from src.envs.uav_comm_env import MiniSystem
import numpy as np
import math
import time
import torch
import csv
import os

# 初始化系统模型
episode_num = EPISODE_NUM
step_num = 100

# 如果指定了加载路径，检查已完成的轮数
existing_episode_count = 0
if LOAD_PATH is not None and os.path.exists(LOAD_PATH):
    csv_path = os.path.join(LOAD_PATH, 'step_num_per_episode.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            existing_episode_count = sum(1 for row in reader)
        print(f"Found {existing_episode_count} existing episodes in {LOAD_PATH}")

project_name = f'uav_bs_fas/{DRL_ALGO}_{REWARD_DESIGN}' if TRAINED_UAV else f'uav_bs_fas/scratch/{DRL_ALGO}_{REWARD_DESIGN}'

# 使用 UAV-FAS 一体化系统
# 无人机搭载：FAS流体天线（唯一发射天线，支持主动波束成形）
# 地面固定：RIS反射单元阵列（64单元，8x8）
system = MiniSystem(
    user_num=2,
    fas_ant_num=12,     # FAS流体天线发射单元数（搭载在无人机上）
    ris_ant_num=64,     # RIS反射单元数（地面固定，8x8 UPA）
    if_dir_link=1,
    if_with_FAS=True,   # 使用FAS
    if_move_users=False, # 用户静止
    if_movements=True,
    reverse_x_y=(False, False),
    if_UAV_pos_state = True,
    reward_design = REWARD_DESIGN,
    project_name = project_name,
    step_num = step_num,
    existing_path = LOAD_PATH,
    num_active_ports=2  # 同时激活2个FAS端口 (Gumbel-Softmax Top-2)
)

if_Theta_fixed = False
if_G_fixed = False
if_BS = False
if_robust = True

# 初始化 Agent 1 (BS波束成形 + FAS反射系数)
agent_1_param_dic = {}
agent_1_param_dic["alpha"] = 0.0001  # Actor学习率
agent_1_param_dic["beta"] = 0.001   # Critic学习率
agent_1_param_dic["input_dims"] = system.get_system_state_dim()
agent_1_param_dic["tau"] = 0.005    # soft update
agent_1_param_dic["batch_size"] = 80  # 批次大小
agent_1_param_dic["n_actions"] = system.get_system_action_dim()  # 38维动作空间
agent_1_param_dic["action_noise_factor"] = 0.3  # 初始噪声
agent_1_param_dic["memory_max_size"] = 500000  # 经验回放池容量
agent_1_param_dic["agent_name"] = "BS_FAS"
agent_1_param_dic["n_discrete_dims"] = 12  # Gumbel-Softmax端口logits维度
agent_1_param_dic["layer1_size"] = 1024
agent_1_param_dic["layer2_size"] = 768
agent_1_param_dic["layer3_size"] = 512
agent_1_param_dic["layer4_size"] = 256

# 初始化 Agent 2 (UAV轨迹控制)
agent_2_param_dic = {}
agent_2_param_dic["alpha"] = 0.0001  # Actor学习率
agent_2_param_dic["beta"] = 0.001   # Critic学习率
agent_2_param_dic["input_dims"] = system.get_uav_local_state_dim()  # 输入维度：18维本地信息
agent_2_param_dic["tau"] = 0.005    # soft update
agent_2_param_dic["batch_size"] = 80  # 批次大小
agent_2_param_dic["n_actions"] = 2  # 输出2维：水平速度[vx,vy] (高度固定50m)
agent_2_param_dic["action_noise_factor"] = 0.3  # 初始噪声
agent_2_param_dic["memory_max_size"] = 500000  # 经验回放池容量
agent_2_param_dic["agent_name"] = "UAV"
agent_2_param_dic["layer1_size"] = 512
agent_2_param_dic["layer2_size"] = 384
agent_2_param_dic["layer3_size"] = 256
agent_2_param_dic["layer4_size"] = 128

if SEEDS is not None:
    torch.manual_seed(SEEDS[0])
    torch.cuda.manual_seed_all(SEEDS[0])

agent_1 = Agent(
    alpha       = agent_1_param_dic["alpha"],
    beta        = agent_1_param_dic["beta"],
    input_dims  = [agent_1_param_dic["input_dims"]],
    tau         = agent_1_param_dic["tau"],
    env         = system,
    batch_size  = agent_1_param_dic["batch_size"],
    layer1_size=agent_1_param_dic["layer1_size"],
    layer2_size=agent_1_param_dic["layer2_size"],
    layer3_size=agent_1_param_dic["layer3_size"],
    layer4_size=agent_1_param_dic["layer4_size"],
    n_actions   = agent_1_param_dic["n_actions"],
    max_size = agent_1_param_dic["memory_max_size"],
    agent_name= agent_1_param_dic["agent_name"],
    n_discrete_dims=agent_1_param_dic["n_discrete_dims"]
) 

if SEEDS is not None:
    torch.manual_seed(SEEDS[-1])
    torch.cuda.manual_seed_all(SEEDS[-1])

agent_2 = Agent(
    alpha       = agent_2_param_dic["alpha"],
    beta        = agent_2_param_dic["beta"],
    input_dims  = [agent_2_param_dic["input_dims"]],
    tau         = agent_2_param_dic["tau"],
    env         = system,
    batch_size  = agent_2_param_dic["batch_size"],
    layer1_size=agent_2_param_dic["layer1_size"],
    layer2_size=agent_2_param_dic["layer2_size"], 
    layer3_size=agent_2_param_dic["layer3_size"],
    layer4_size=agent_2_param_dic["layer4_size"],
    n_actions   = agent_2_param_dic["n_actions"],
    max_size = agent_2_param_dic["memory_max_size"],
    agent_name= agent_2_param_dic["agent_name"]
) 

# 打印系统信息
meta_dic = {}
print("***********************UAV-FAS 系统信息******************************")
print("folder_name:     "+str(system.data_manager.store_path))
meta_dic['folder_name'] = system.data_manager.store_path
print("user_num:        "+str(system.user_num))
meta_dic['user_num'] = system.user_num
print("if_dir:          "+str(system.if_dir_link))
meta_dic['if_dir_link'] = system.if_dir_link
print("if_with_FAS:     "+str(system.if_with_FAS))
meta_dic['if_with_FAS'] = system.if_with_FAS
print("if_user_m:       "+str(system.if_move_users))
meta_dic['if_move_users'] = system.if_move_users
print("FAS_ant_num:     "+str(system.UAV_FAS.fas_num_ports))
meta_dic['fas_ant_num'] = system.UAV_FAS.fas_num_ports
print("if_movements:    "+str(system.if_movements))
meta_dic['system_if_movements'] = system.if_movements
print("reverse_x_y:     "+str(system.reverse_x_y))
meta_dic['system_reverse_x_y'] = system.reverse_x_y
print("if_UAV_pos_state:"+str(system.if_UAV_pos_state))
meta_dic['if_UAV_pos_state'] = system.if_UAV_pos_state

print("ep_num:          "+str(episode_num))
meta_dic['episode_num'] = episode_num
print("step_num:        "+str(step_num))
meta_dic['step_num'] = step_num

print("***********************Agent 1 (BS+FAS) 信息******************************")
tplt = "{0:{2}^20}\t{1:{2}^20}"
for i in agent_1_param_dic:
    parm = agent_1_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_1"] = agent_1_param_dic

print("***********************Agent 2 (UAV) 信息******************************")
for i in agent_2_param_dic:
    parm = agent_2_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_2"] = agent_2_param_dic

system.data_manager.save_meta_data(meta_dic)

# 如果指定了加载路径，加载已有模型
if LOAD_PATH is not None and existing_episode_count > 0:
    print(f"Loading models from {LOAD_PATH}...")
    if DRL_ALGO == 'td3':
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
    elif DRL_ALGO == 'ddpg':
        agent_1.load_models(
            load_file_actor=os.path.join(LOAD_PATH, 'Actor_BS_FAS_ddpg'),
            load_file_critic=os.path.join(LOAD_PATH, 'Critic_BS_FAS_ddpg')
        )
        agent_2.load_models(
            load_file_actor=os.path.join(LOAD_PATH, 'Actor_UAV_ddpg'),
            load_file_critic=os.path.join(LOAD_PATH, 'Critic_UAV_ddpg')
        )
    print(f"Models loaded successfully!")

episode_cnt = existing_episode_count
total_episodes = EPISODE_NUM

print("***********************训练开始******************************")
print(f"Starting from episode {episode_cnt}, training to {total_episodes} episodes")

# 训练日志
episode_rewards = []
episode_capacities = []
best_reward = -float('inf')

while episode_cnt < total_episodes:
    system.reset()
    system.training = True  # 训练模式: Gumbel-Softmax可微分
    step_cnt = 0
    score_per_ep = 0

    if if_robust:
        tmp = system.observe()
        z = np.random.normal(size=len(tmp))
        observersion_1 = list(np.array(tmp) + 0.6 * 1e-7 * z)
    else:
        observersion_1 = system.observe()

    observersion_2 = system.observe_uav_local()  # 获取无人机本地局部状态

    while step_cnt < step_num:
        step_cnt += 1

        # 自适应噪声: 缓慢衰减，保持探索
        noise_scale_1 = agent_1_param_dic["action_noise_factor"] * max(0.3, 1 - episode_cnt / episode_num)
        noise_scale_2 = agent_2_param_dic["action_noise_factor"] * max(0.3, 1 - episode_cnt / episode_num)

        action_1 = agent_1.choose_action(observersion_1, greedy=noise_scale_1)
        action_2 = agent_2.choose_action(observersion_2, greedy=noise_scale_2)

        # 动作解析：38维 = 13维FAS (12端口选择 + 1增益) + 25维RIS (1β + 24相位)
        # G: 前12维为端口选择softmax，第13维为F增益
        # Phi: 第1维为β，后24维为RIS相位

        if if_Theta_fixed:
            action_1[13:] = np.zeros(26)  # 固定RIS相位为零

        # 执行动作 (FAS-only mode)
        # 动作布局: action_1[0:12]=端口, [12]=F增益, [13]=β, [14]=η(jam_ratio), [15:39]=相位
        new_state_1, reward, done, info = system.step(
            action_0=action_2[0],  # vx速度
            action_1=action_2[1],  # vy速度
            action_2=0,  # 高度固定50m
            G=action_1[0:13],  # 13维: 12端口选择 + 1 F增益
            Phi=action_1[13:39],  # 26维RIS: 1维β + 1维η + 24维相位
            set_pos_x=action_2[0],
            set_pos_y=action_2[1]
        )
        new_state_2 = system.observe_uav_local()  # 获取无人机本地局部状态

        score_per_ep += reward
        
        agent_1.remember(observersion_1, action_1, reward, new_state_1, int(done))
        agent_2.remember(observersion_2, action_2, reward, new_state_2, int(done))

        agent_1.learn()
        # 双Agent稳定性: Agent 2每2步学习1次，降低非稳态影响
        if not TRAINED_UAV and step_cnt % 2 == 0:
            agent_2.learn()

        observersion_1 = new_state_1
        observersion_2 = new_state_2

        if done == True:
            break

    system.data_manager.save_file(episode_cnt=episode_cnt)

    # 记录训练指标 (必须在reset之前读取capacity)
    avg_cap = np.mean([u.capacity for u in system.user_list])
    episode_rewards.append(score_per_ep)
    episode_capacities.append(avg_cap)

    system.reset()

    # 干扰相位课程学习衰减: 逐渐减少对准窃听者的权重，让Agent学会自主控制
    system.jam_align_weight = max(
        system.jam_align_min,
        system.jam_align_weight * system.jam_align_decay
    )

    # 打印训练进度
    if episode_cnt % 10 == 0:
        avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_cap_10 = np.mean(episode_capacities[-10:]) if len(episode_capacities) >= 10 else np.mean(episode_capacities)
        print(f"ep: {episode_cnt:4d} | reward: {score_per_ep:8.3f} | avg10: {avg_reward_10:8.3f} | cap: {avg_cap:.4f} | jam_align: {system.jam_align_weight:.3f}")

    episode_cnt += 1

    if episode_cnt % 50 == 0:
        agent_1.save_models()
        agent_2.save_models()

agent_1.save_models()
agent_2.save_models()

# 保存训练曲线数据
np.savetxt(system.data_manager.store_path + '/training_rewards.csv',
           np.array(episode_rewards), delimiter=',', header='reward', comments='')
np.savetxt(system.data_manager.store_path + '/training_capacities.csv',
           np.array(episode_capacities), delimiter=',', header='capacity', comments='')

print("***********************训练完成******************************")
print(f"最终10轮平均奖励: {np.mean(episode_rewards[-10:]):.4f}")
print(f"最终10轮平均容量: {np.mean(episode_capacities[-10:]):.6f}")