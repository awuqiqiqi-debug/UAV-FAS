"""自动续训脚本：从最新检查点训练到1000轮，自动处理中断"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.argv = ['train_to_1000.py', '--drl', 'td3', '--reward', 'see']

import shutil
import csv
import glob
from src.agents.td3_agent import Agent
from src.envs.uav_comm_env_legacy import MiniSystem
import numpy as np
import math
import torch

TOTAL_EPISODES = 1000
BASE_DIR = 'data/storage/scratch'
TARGET_NAME = 'td3_see_1000'

def count_episodes(path):
    if not os.path.exists(path):
        return 0
    csv_path = os.path.join(path, 'step_num_per_episode.csv')
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, 'r') as f:
        return sum(1 for _ in csv.reader(f))

def find_load_path():
    """Find the most recent td3_see_1000* directory"""
    candidates = sorted(glob.glob(os.path.join(BASE_DIR, 'td3_see_1000*')),
                        key=os.path.getmtime, reverse=True)
    for c in candidates:
        if count_episodes(c) > 0:
            return c
    return None

def train_chunk(start_ep, load_path, store_path):
    """Train one chunk. Returns (last_episode, completed)"""
    system = MiniSystem(
        user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
        if_with_RIS=True, if_move_users=True, if_movements=True,
        reverse_x_y=(False, False), if_UAV_pos_state=True,
        reward_design='see', project_name=None, step_num=100,
        existing_path=load_path)

    agent_1 = Agent(alpha=1e-4, beta=1e-3, input_dims=[system.get_system_state_dim()], tau=1e-3,
        env=system, batch_size=64, layer1_size=800, layer2_size=600,
        layer3_size=512, layer4_size=256, n_actions=17, max_size=100000,
        agent_name="G_and_Phi")

    agent_2 = Agent(alpha=1e-4, beta=1e-3, input_dims=[12], tau=1e-3,
        env=system, batch_size=64, layer1_size=400, layer2_size=300,
        layer3_size=256, layer4_size=128, n_actions=2, max_size=100000,
        agent_name="UAV")

    # Load models
    agent_1.load_models(
        load_file_actor=os.path.join(load_path, 'Actor_G_and_Phi_TD3'),
        load_file_critic_1=os.path.join(load_path, 'Critic_1_G_and_Phi_TD3'),
        load_file_critic_2=os.path.join(load_path, 'Critic_2_G_and_Phi_TD3'))
    agent_2.load_models(
        load_file_actor=os.path.join(load_path, 'Actor_UAV_TD3'),
        load_file_critic_1=os.path.join(load_path, 'Critic_1_UAV_TD3'),
        load_file_critic_2=os.path.join(load_path, 'Critic_2_UAV_TD3'))

    episode_cnt = start_ep
    step_num = 100

    while episode_cnt < TOTAL_EPISODES:
        system.reset()
        step_cnt = 0
        score_per_ep = 0

        tmp = system.observe()
        z = np.random.normal(size=len(tmp))
        obs1 = list(np.array(tmp) + 0.6 * 1e-7 * z)
        obs2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
        for user in system.user_list:
            obs2 += list(user.coordinate)

        while step_cnt < step_num:
            step_cnt += 1
            g1 = 0.1 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
            g2 = 0.5 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
            a1 = agent_1.choose_action(obs1, greedy=g1)
            a2 = agent_2.choose_action(obs2, greedy=g2)

            n_G = 2 * system.UAV.bs_ant_num * system.user_num
            F_dummy = list(np.zeros(2 * system.UAV.fas_num_ports * system.user_num))
            new_s1, reward, done, _ = system.step(
                action_0=a2[0], action_1=a2[1],
                G=a1[:n_G], F=F_dummy,
                set_pos_x=a2[0], set_pos_y=a2[1],
                fas_port_signal=a1[16])
            new_s2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
            for user in system.user_list:
                new_s2 += list(user.coordinate)

            score_per_ep += reward
            agent_1.remember(obs1, a1, reward, new_s1, int(done))
            agent_2.remember(obs2, a2, reward, new_s2, int(done))
            agent_1.learn()
            agent_2.learn()
            obs1, obs2 = new_s1, new_s2
            if done:
                break

        system.data_manager.save_file(episode_cnt=episode_cnt)
        system.reset()
        print(f"ep_num: {episode_cnt}   ep_score:  {score_per_ep}", flush=True)
        episode_cnt += 1

        if episode_cnt % 10 == 0:
            agent_1.save_models()
            agent_2.save_models()

    agent_1.save_models()
    agent_2.save_models()
    return episode_cnt, episode_cnt >= TOTAL_EPISODES


# === Main loop ===
load_path = find_load_path()
if load_path:
    existing = count_episodes(load_path)
    print(f"Found existing training: {load_path} ({existing} episodes)")
else:
    existing = 0
    print("No existing training found, starting fresh")

# The data_manager will create a new directory each time it's called.
# We need to manually manage the target directory.
# Strategy: always write to the same directory by using existing_path

while existing < TOTAL_EPISODES:
    if load_path is None:
        # First run: let MiniSystem create the directory
        print(f"\n=== Starting fresh training ===")
        system = MiniSystem(
            user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
            if_with_RIS=True, if_move_users=True, if_movements=True,
            reverse_x_y=(False, False), if_UAV_pos_state=True,
            reward_design='see', project_name=f'scratch/{TARGET_NAME}', step_num=100)

        agent_1 = Agent(alpha=1e-4, beta=1e-3, input_dims=[system.get_system_state_dim()], tau=1e-3,
            env=system, batch_size=64, layer1_size=800, layer2_size=600,
            layer3_size=512, layer4_size=256, n_actions=16, max_size=100000,
            agent_name="G_and_Phi")
        agent_2 = Agent(alpha=1e-4, beta=1e-3, input_dims=[12], tau=1e-3,
            env=system, batch_size=64, layer1_size=400, layer2_size=300,
            layer3_size=256, layer4_size=128, n_actions=3, max_size=100000,
            agent_name="UAV")

        load_path = system.data_manager.store_path
        existing = 0
    else:
        # Resume: but data_manager creates new dir, so we need special handling
        existing = count_episodes(load_path)
        print(f"\n=== Resuming from {load_path} ({existing} episodes) ===")

    if existing >= TOTAL_EPISODES:
        print("Already at target!")
        break

    # For resume runs, we need to write to the SAME directory
    # Use a wrapper that bypasses data_manager's auto-increment
    try:
        if existing == 0:
            # Fresh start: use the normal training loop
            episode_cnt = 0
            step_num = 100
            while episode_cnt < TOTAL_EPISODES:
                system.reset()
                step_cnt = 0
                score_per_ep = 0
                tmp = system.observe()
                z = np.random.normal(size=len(tmp))
                obs1 = list(np.array(tmp) + 0.6 * 1e-7 * z)
                obs2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
                for user in system.user_list:
                    obs2 += list(user.coordinate)
                while step_cnt < step_num:
                    step_cnt += 1
                    g1 = 0.1 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
                    g2 = 0.5 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
                    a1 = agent_1.choose_action(obs1, greedy=g1)
                    a2 = agent_2.choose_action(obs2, greedy=g2)
                    F_dummy = list(np.zeros(2 * system.UAV.fas_num_ports * system.user_num))
                    new_s1, reward, done, _ = system.step(
                        action_0=a2[0], action_1=a2[1],
                        G=list(a1), F=F_dummy,
                        set_pos_x=a2[0], set_pos_y=a2[1],
                fas_port_signal=a1[16])
                    new_s2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
                    for user in system.user_list:
                        new_s2 += list(user.coordinate)
                    score_per_ep += reward
                    agent_1.remember(obs1, a1, reward, new_s1, int(done))
                    agent_2.remember(obs2, a2, reward, new_s2, int(done))
                    agent_1.learn()
                    agent_2.learn()
                    obs1, obs2 = new_s1, new_s2
                    if done:
                        break
                system.data_manager.save_file(episode_cnt=episode_cnt)
                system.reset()
                print(f"ep_num: {episode_cnt}   ep_score:  {score_per_ep}", flush=True)
                episode_cnt += 1
                if episode_cnt % 10 == 0:
                    agent_1.save_models()
                    agent_2.save_models()
            agent_1.save_models()
            agent_2.save_models()
            existing = episode_cnt
        else:
            # Resume: manually save to existing directory
            # Create a temporary MiniSystem, then override data_manager path
            system = MiniSystem(
                user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
                if_with_RIS=True, if_move_users=True, if_movements=True,
                reverse_x_y=(False, False), if_UAV_pos_state=True,
                reward_design='see', project_name=None, step_num=100)
            # Override store path to write to existing directory
            system.data_manager.store_path = load_path

            agent_1 = Agent(alpha=1e-4, beta=1e-3, input_dims=[system.get_system_state_dim()], tau=1e-3,
                env=system, batch_size=64, layer1_size=800, layer2_size=600,
                layer3_size=512, layer4_size=256, n_actions=16, max_size=100000,
                agent_name="G_and_Phi")
            agent_2 = Agent(alpha=1e-4, beta=1e-3, input_dims=[12], tau=1e-3,
                env=system, batch_size=64, layer1_size=400, layer2_size=300,
                layer3_size=256, layer4_size=128, n_actions=3, max_size=100000,
                agent_name="UAV")

            agent_1.load_models(
                load_file_actor=os.path.join(load_path, 'Actor_G_and_Phi_TD3'),
                load_file_critic_1=os.path.join(load_path, 'Critic_1_G_and_Phi_TD3'),
                load_file_critic_2=os.path.join(load_path, 'Critic_2_G_and_Phi_TD3'))
            agent_2.load_models(
                load_file_actor=os.path.join(load_path, 'Actor_UAV_TD3'),
                load_file_critic_1=os.path.join(load_path, 'Critic_1_UAV_TD3'),
                load_file_critic_2=os.path.join(load_path, 'Critic_2_UAV_TD3'))

            episode_cnt = existing
            step_num = 100
            while episode_cnt < TOTAL_EPISODES:
                system.reset()
                step_cnt = 0
                score_per_ep = 0
                tmp = system.observe()
                z = np.random.normal(size=len(tmp))
                obs1 = list(np.array(tmp) + 0.6 * 1e-7 * z)
                obs2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
                for user in system.user_list:
                    obs2 += list(user.coordinate)
                while step_cnt < step_num:
                    step_cnt += 1
                    g1 = 0.1 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
                    g2 = 0.5 * math.pow((1 - episode_cnt / TOTAL_EPISODES), 2)
                    a1 = agent_1.choose_action(obs1, greedy=g1)
                    a2 = agent_2.choose_action(obs2, greedy=g2)
                    F_dummy = list(np.zeros(2 * system.UAV.fas_num_ports * system.user_num))
                    new_s1, reward, done, _ = system.step(
                        action_0=a2[0], action_1=a2[1],
                        G=list(a1), F=F_dummy,
                        set_pos_x=a2[0], set_pos_y=a2[1],
                fas_port_signal=a1[16])
                    new_s2 = list(system.UAV.coordinate) + list(system.RIS.coordinate)
                    for user in system.user_list:
                        new_s2 += list(user.coordinate)
                    score_per_ep += reward
                    agent_1.remember(obs1, a1, reward, new_s1, int(done))
                    agent_2.remember(obs2, a2, reward, new_s2, int(done))
                    agent_1.learn()
                    agent_2.learn()
                    obs1, obs2 = new_s1, new_s2
                    if done:
                        break
                system.data_manager.save_file(episode_cnt=episode_cnt)
                system.reset()
                print(f"ep_num: {episode_cnt}   ep_score:  {score_per_ep}", flush=True)
                episode_cnt += 1
                if episode_cnt % 10 == 0:
                    agent_1.save_models()
                    agent_2.save_models()
            agent_1.save_models()
            agent_2.save_models()
            existing = episode_cnt

    except Exception as e:
        print(f"\nTraining interrupted: {e}")
        existing = count_episodes(load_path)
        print(f"Saved up to episode {existing}, will resume...")

    if existing >= TOTAL_EPISODES:
        print(f"\n=== Training complete! {existing} episodes ===")
        break
