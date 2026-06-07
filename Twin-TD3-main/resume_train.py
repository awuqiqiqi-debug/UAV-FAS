import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.argv = ['resume_train.py', '--drl', 'td3', '--reward', 'see']

from td3 import Agent
from env import MiniSystem

LOAD_PATH = 'data/storage/scratch/td3_see_5'
STORE_PATH = LOAD_PATH  # Write directly to existing directory
import numpy as np
import math
import torch
import csv

DRL_ALGO = 'td3'
REWARD_DESIGN = 'see'
TOTAL_EPISODES = 500
DATA_PATH = 'data/storage/scratch/td3_see_5'

# Count existing episodes from LOAD_PATH
existing_count = 0
csv_path = os.path.join(LOAD_PATH, 'step_num_per_episode.csv')
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        existing_count = sum(1 for row in reader)

print(f"Found {existing_count} existing episodes in {LOAD_PATH}, training to {TOTAL_EPISODES}")

if existing_count >= TOTAL_EPISODES:
    print("Already complete!")
    sys.exit(0)

# Init system (creates new data dir, that's ok)
system = MiniSystem(
    user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
    if_with_RIS=True, if_move_users=True, if_movements=True,
    reverse_x_y=(False, False), if_UAV_pos_state=True,
    reward_design=REWARD_DESIGN, project_name='scratch/td3_see_resume', step_num=100)

# Init agents
agent_1 = Agent(alpha=1e-4, beta=1e-3, input_dims=[27], tau=1e-3,
    env=system, batch_size=64, layer1_size=800, layer2_size=600,
    layer3_size=512, layer4_size=256, n_actions=20, max_size=50000,
    agent_name="G_and_Phi")

agent_2 = Agent(alpha=1e-4, beta=1e-3, input_dims=[12], tau=1e-3,
    env=system, batch_size=64, layer1_size=400, layer2_size=300,
    layer3_size=256, layer4_size=128, n_actions=2, max_size=50000,
    agent_name="UAV")

# Load models from LOAD_PATH
if DRL_ALGO == 'td3':
    agent_1.load_models(
        load_file_actor=os.path.join(LOAD_PATH, 'Actor_G_and_Phi_TD3'),
        load_file_critic_1=os.path.join(LOAD_PATH, 'Critic_1_G_and_Phi_TD3'),
        load_file_critic_2=os.path.join(LOAD_PATH, 'Critic_2_G_and_Phi_TD3'))
    agent_2.load_models(
        load_file_actor=os.path.join(LOAD_PATH, 'Actor_UAV_TD3'),
        load_file_critic_1=os.path.join(LOAD_PATH, 'Critic_1_UAV_TD3'),
        load_file_critic_2=os.path.join(LOAD_PATH, 'Critic_2_UAV_TD3'))
print("Models loaded!")

episode_cnt = existing_count
step_num = 100

while episode_cnt < TOTAL_EPISODES:
    system.reset()
    step_cnt = 0
    score_per_ep = 0

    tmp = system.observe()
    z = np.random.normal(size=len(tmp))
    observersion_1 = list(np.array(tmp) + 0.6 * 1e-7 * z)
    observersion_2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
    for user in system.user_list:
        observersion_2 += list(user.coordinate)

    while step_cnt < step_num:
        step_cnt += 1
        greedy_1 = 0.1 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
        greedy_2 = 0.5 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
        action_1 = agent_1.choose_action(observersion_1, greedy=greedy_1)
        action_2 = agent_2.choose_action(observersion_2, greedy=greedy_2)

        new_state_1, reward, done, info = system.step(
            action_0=action_2[0], action_1=action_2[1],
            G=action_1[0:16], Phi=action_1[16:],
            set_pos_x=action_2[0], set_pos_y=action_2[1])
        new_state_2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
        for user in system.user_list:
            new_state_2 += list(user.coordinate)

        score_per_ep += reward
        agent_1.remember(observersion_1, action_1, reward, new_state_1, int(done))
        agent_2.remember(observersion_2, action_2, reward, new_state_2, int(done))
        agent_1.learn()
        agent_2.learn()
        observersion_1 = new_state_1
        observersion_2 = new_state_2
        if done:
            break

    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()
    print(f"ep_num: {episode_cnt}   ep_score:  {score_per_ep}")
    episode_cnt += 1

    if episode_cnt % 10 == 0:
        agent_1.save_models()
        agent_2.save_models()

agent_1.save_models()
agent_2.save_models()
print("Training complete!")
