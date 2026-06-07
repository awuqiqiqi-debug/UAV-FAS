import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.io import loadmat, savemat
import pandas as pd
import os
import copy
import math
import csv

import argparse

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, required = False, default=None, help='the path where the training/simulation data is stored')
parser.add_argument('--ep-num', type = int, required = False, default=300, help='total number of episodes')


# extract argument
args = parser.parse_args()
STORE_PATH = args.path
EP_NUM = args.ep_num


######################################################
# new for energy 
# energy related parameters of rotary-wing UAV
# based on Energy Minimization in Internet-of-Things System Based on Rotary-Wing UAV
P_i = 790.6715
P_0 = 580.65
U2_tip = (200) ** 2
s = 0.05
d_0 = 0.3
p = 1.225
A = 0.79
delta_time = 0.1 #0.1/1000 #0.1ms

# add ons hover veloctiy
# based on https://www.intechopen.com/chapters/57483
m = 1.3 # mass: assume 1.3kg https://www.droneblog.com/average-weights-of-common-types-of-drones/#:~:text=In%20most%20cases%2C%20toy%20drones,What%20is%20this%3F
g = 9.81 # gravity
T = m * g # thrust
v_0 = (T / (A * 2 * p)) ** 0.5

def get_energy_consumption(v_t):
    '''
    arg
    1) v_t = displacement per time slot
    '''
    energy_1 = P_0 \
                + 3 * P_0 * (abs(v_t)) ** 2 / U2_tip \
                + 0.5 * d_0 * p * s * A * (abs(v_t))**3
    
    energy_2 = P_i * ((
                    (1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5 \
                    - (abs(v_t) ** 2) / (2 * (v_0 **2)) \
                ) ** 0.5)
    
    energy = delta_time * (energy_1 + energy_2)
    return energy 

ENERGY_MIN = get_energy_consumption(0.25)
ENERGY_MAX = get_energy_consumption(0)

######################################################


# modified from data_manager.py
init_data_file = 'data/init_location.xlsx'
def read_init_location(entity_type = 'user', index = 0):
    if entity_type == 'user' or 'attacker' or 'RIS' or 'RIS_norm_vec' or 'UAV':
        return np.array([\
        pd.read_excel(init_data_file, sheet_name=entity_type)['x'][index],\
        pd.read_excel(init_data_file, sheet_name=entity_type)['y'][index],\
        pd.read_excel(init_data_file, sheet_name=entity_type)['z'][index]])
    else:
        return None


# load and plot everything
class LoadAndPlot(object):
    """
    load date and plot 2022-07-22 16_16_26
    """
    def __init__(self, store_path, \
                       user_num = 2, attacker_num = 1, RIS_ant_num = 4, \
                       ep_num = EP_NUM, step_num = 100): # RIS_ant_num = 16 (not true)

        self.color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        self.store_path = store_path + '//'
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.RIS_ant_num = RIS_ant_num
        self.ep_num = ep_num
        self.step_num = step_num

        self.all_steps = self.load_all_steps()


    def load_one_ep(self, file_name):
        m = loadmat(self.store_path + file_name)
        return m


    def load_all_steps(self):
        result_dic = {}
        result_dic.update({'reward':[]})

        result_dic.update({'user_capacity':[]})
        for i in range(self.user_num):
            result_dic['user_capacity'].append([])

        result_dic.update({'secure_capacity':[]})
        for i in range(self.user_num):
            result_dic['secure_capacity'].append([])

        result_dic.update({'attaker_capacity':[]})
        for i in range(self.attacker_num):
            result_dic['attaker_capacity'].append([])
        
        result_dic.update({'RIS_elements':[]})
        for i in range(self.RIS_ant_num):
            result_dic['RIS_elements'].append([])

        for ep_cnt in range(self.ep_num):
            # 检查文件是否存在
            file_path = self.store_path + "simulation_result_ep_" + str(ep_cnt) + ".mat"
            if not os.path.exists(file_path):
                print(f"跳过缺失的文件: {file_path}")
                continue
            
            mat_ep = self.load_one_ep("simulation_result_ep_" + str(ep_cnt) + ".mat")

            one_ep_reward = mat_ep["result_" + str(ep_cnt)]["reward"][0][0]
            result_dic['reward'] += list(one_ep_reward[:, 0])

            one_ep_user_capacity = mat_ep["result_" + str(ep_cnt)]["user_capacity"][0][0]
            for i in range(self.user_num):
                result_dic['user_capacity'][i] += list(one_ep_user_capacity[:, i])
            
            one_ep_secure_capacity = mat_ep["result_" + str(ep_cnt)]["secure_capacity"][0][0]
            for i in range(self.user_num):
                result_dic['secure_capacity'][i] += list(one_ep_secure_capacity[:, i])
            
            one_ep_attaker_capacity = mat_ep["result_" + str(ep_cnt)]["attaker_capacity"][0][0]
            for i in range(self.attacker_num):
                result_dic['attaker_capacity'][i] += list(one_ep_attaker_capacity[:, i])

            one_ep_RIS_first_element = mat_ep["result_" + str(ep_cnt)]["reflecting_coefficient"][0][0]
            for i in range(self.RIS_ant_num):
                result_dic['RIS_elements'][i] += list(one_ep_RIS_first_element[:, i])

        return result_dic


    def plot(self):
        """
        plot result
        b--blue c--cyan(青色） g--green k--black m--magenta（紫红色） r--red w--white y--yellow
        """
        if not os.path.exists(self.store_path + 'plot'):
            os.makedirs(self.store_path + 'plot')
            os.makedirs(self.store_path + 'plot/RIS')

        # 全局样式设置
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13,
            'figure.titlesize': 20,
            'lines.linewidth': 2.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        })

        color_list = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']

        ###############################
        # read step counts per episode
        ###############################
        step_num_per_episode = []
        with open(self.store_path + 'step_num_per_episode.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                step_num_per_episode.append(int(row[0]))

        ###############################
        # 计算每轮平均值（后续多个图共用）
        ###############################
        # SSR per episode
        sum_secrecy_rate = np.array(self.all_steps['secure_capacity'])
        sum_secrecy_rate = np.sum(sum_secrecy_rate, axis=0)
        average_sum_secrecy_rate = []
        ssr = []
        j = 0
        for i in range(self.ep_num):
            if i < len(step_num_per_episode):
                ssr_one_episode = sum_secrecy_rate[j:j+step_num_per_episode[i]]
                j = j + step_num_per_episode[i]
            else:
                ssr_one_episode = np.array([0])
            ssr.append(ssr_one_episode)
            try:
                _ = sum(ssr_one_episode) / len(ssr_one_episode)
            except:
                _ = 0
            average_sum_secrecy_rate.append(_)

        # User capacity per episode (平均)
        avg_user_cap_per_ep = []
        for ep_i in range(self.ep_num):
            ep_avgs = []
            for u_i in range(self.user_num):
                # 计算该轮该用户的平均容量
                start_idx = sum(step_num_per_episode[:ep_i]) if ep_i < len(step_num_per_episode) else 0
                end_idx = start_idx + step_num_per_episode[ep_i] if ep_i < len(step_num_per_episode) else start_idx
                if end_idx <= len(self.all_steps['user_capacity'][u_i]) and start_idx < end_idx:
                    ep_avgs.append(np.mean(self.all_steps['user_capacity'][u_i][start_idx:end_idx]))
                else:
                    ep_avgs.append(0)
            avg_user_cap_per_ep.append(ep_avgs)

        # Secure capacity per episode (平均)
        avg_sec_cap_per_ep = []
        for ep_i in range(self.ep_num):
            ep_avgs = []
            for u_i in range(self.user_num):
                start_idx = sum(step_num_per_episode[:ep_i]) if ep_i < len(step_num_per_episode) else 0
                end_idx = start_idx + step_num_per_episode[ep_i] if ep_i < len(step_num_per_episode) else start_idx
                if end_idx <= len(self.all_steps['secure_capacity'][u_i]) and start_idx < end_idx:
                    ep_avgs.append(np.mean(self.all_steps['secure_capacity'][u_i][start_idx:end_idx]))
                else:
                    ep_avgs.append(0)
            avg_sec_cap_per_ep.append(ep_avgs)

        # Attacker capacity per episode (平均)
        avg_att_cap_per_ep = []
        for ep_i in range(self.ep_num):
            ep_avgs = []
            for a_i in range(self.attacker_num):
                start_idx = sum(step_num_per_episode[:ep_i]) if ep_i < len(step_num_per_episode) else 0
                end_idx = start_idx + step_num_per_episode[ep_i] if ep_i < len(step_num_per_episode) else start_idx
                if end_idx <= len(self.all_steps['attaker_capacity'][a_i]) and start_idx < end_idx:
                    ep_avgs.append(np.mean(self.all_steps['attaker_capacity'][a_i][start_idx:end_idx]))
                else:
                    ep_avgs.append(0)
            avg_att_cap_per_ep.append(ep_avgs)

        avg_user_cap_per_ep = np.array(avg_user_cap_per_ep)
        avg_sec_cap_per_ep = np.array(avg_sec_cap_per_ep)
        avg_att_cap_per_ep = np.array(avg_att_cap_per_ep)

        ###############################
        # plot reward with moving average
        ###############################
        fig, ax = plt.subplots(figsize=(12, 6))
        reward_data = np.array(self.all_steps['reward'])
        # 按轮次计算平均奖励
        avg_reward_per_ep = []
        j = 0
        for i in range(self.ep_num):
            if i < len(step_num_per_episode):
                ep_reward = reward_data[j:j+step_num_per_episode[i]]
                j += step_num_per_episode[i]
                avg_reward_per_ep.append(np.mean(ep_reward))
            else:
                avg_reward_per_ep.append(0)
        avg_reward_per_ep = np.array(avg_reward_per_ep)

        ax.plot(range(len(avg_reward_per_ep)), avg_reward_per_ep, color='#90CAF9', alpha=0.4, linewidth=1.0, label='Per-episode')
        # 滑动平均
        window = min(20, len(avg_reward_per_ep) // 3)
        if window >= 2 and len(avg_reward_per_ep) >= window:
            smooth = np.convolve(avg_reward_per_ep, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(avg_reward_per_ep)), smooth, color='#1565C0', linewidth=2.5, label=f'{window}-ep Moving Avg')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.set_title("Training Reward per Episode")
        ax.legend(loc='lower right', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/reward.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


        ###############################
        # plot average sum secrecy rate (SSR) - per episode with smoothing
        ###############################
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(average_sum_secrecy_rate)), average_sum_secrecy_rate, color='#E3F2FD', alpha=0.5, linewidth=1.0)
        window_ssr = min(20, len(average_sum_secrecy_rate) // 3)
        if window_ssr >= 2 and len(average_sum_secrecy_rate) >= window_ssr:
            ssr_smooth = np.convolve(average_sum_secrecy_rate, np.ones(window_ssr)/window_ssr, mode='valid')
            ax.plot(range(window_ssr-1, len(average_sum_secrecy_rate)), ssr_smooth, color='#0D47A1', linewidth=2.5, label=f'{window_ssr}-ep Moving Avg')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Sum Secrecy Rate (bits/s/Hz)")
        ax.set_title("Average Sum Secrecy Rate (SSR) per Episode")
        ax.legend(loc='lower right', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/average_sum_secrecy_rate.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        print()
        print('###########################################################')
        print('Metrics\t\t\tLast Episode\tMax Values Reached')
        print('###########################################################')
        print('SSR (bits/s/Hz)\t\t{:.2f}\t\t{:.2f}'.format(average_sum_secrecy_rate[-1], max(average_sum_secrecy_rate)))


        ###############################
        # plot secrecy energy efficiency (SEE) - per episode with smoothing
        ###############################
        # get init location
        init_uav_coord = read_init_location(entity_type='UAV')
        init_user_coord_0 = read_init_location(entity_type='user', index=0)
        init_user_coord_1 = read_init_location(entity_type='user', index=1)

        ep_num = EP_NUM
        energies = []
        for i in range(ep_num):
            filename = f'simulation_result_ep_{i}.mat'
            filename = os.path.join(STORE_PATH, filename)
            if not os.path.exists(filename):
                energies.append([0])
                continue
            data = loadmat(filename)

            energies_one_episode = []
            uav_movt = data[f'result_{i}'][0][0][-1]
            for j in range(uav_movt.shape[0]):
                move_x = uav_movt[j][0]
                move_y = uav_movt[j][1]
                v_t = (move_x ** 2 + move_y ** 2) ** 0.5
                energy = get_energy_consumption(v_t / delta_time)
                energies_one_episode.append(energy)
            energies.append(energies_one_episode)

        average_see = []
        for ssr_one_episode, energies_one_episode in zip(ssr, energies):
            ssr_one_episode = ssr_one_episode[:len(energies_one_episode)]
            energies_one_episode = energies_one_episode[:len(ssr_one_episode)]
            try:
                see = np.array(ssr_one_episode) / np.array(energies_one_episode)
                average_see.append(sum(see)/len(see))
            except:
                average_see.append(0)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(average_see)), average_see, color='#E8F5E9', alpha=0.5, linewidth=1.0)
        window_see = min(20, len(average_see) // 3)
        if window_see >= 2 and len(average_see) >= window_see:
            see_smooth = np.convolve(average_see, np.ones(window_see)/window_see, mode='valid')
            ax.plot(range(window_see-1, len(average_see)), see_smooth, color='#1B5E20', linewidth=2.5, label=f'{window_see}-ep Moving Avg')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average SEE (bits/s/Hz/kJ)")
        ax.set_title("Average Secrecy Energy Efficiency (SEE) per Episode")
        ax.legend(loc='lower right', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/average_secrecy_energy_efficiency.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        print('Energy (kJ)\t\t{:.2f}\t\t{:.2f}'.format(sum(energies[-1])/1000, sum(energies[np.argmax(average_see)])/1000))
        print('SEE (bits/s/Hz/kJ)\t{:.2f}\t\t{:.2f}'.format(average_see[-1]*1000, max(average_see)*1000))
        print('\nThe final performance is evalulated based on the Last Episode (where exploration=0)\n')


        ###############################
        # plot secure capacity - per episode average (not all steps!)
        ###############################
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(self.user_num):
            ax.plot(range(len(avg_sec_cap_per_ep[:, i])), avg_sec_cap_per_ep[:, i],
                    color=color_list[i], alpha=0.35, linewidth=1.0)
            # 滑动平均
            w = min(20, len(avg_sec_cap_per_ep) // 3)
            if w >= 2 and len(avg_sec_cap_per_ep) >= w:
                smooth = np.convolve(avg_sec_cap_per_ep[:, i], np.ones(w)/w, mode='valid')
                ax.plot(range(w-1, len(avg_sec_cap_per_ep)), smooth,
                        color=color_list[i], linewidth=2.5, label=f'User {i}')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Secure Capacity (bits/s/Hz)")
        ax.set_title("Secure Capacity per Episode (Averaged)")
        ax.legend(loc='best', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/secure_capacity.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


        ###############################
        # plot user capacity - per episode average
        ###############################
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(self.user_num):
            ax.plot(range(len(avg_user_cap_per_ep[:, i])), avg_user_cap_per_ep[:, i],
                    color=color_list[i], alpha=0.35, linewidth=1.0)
            w = min(20, len(avg_user_cap_per_ep) // 3)
            if w >= 2 and len(avg_user_cap_per_ep) >= w:
                smooth = np.convolve(avg_user_cap_per_ep[:, i], np.ones(w)/w, mode='valid')
                ax.plot(range(w-1, len(avg_user_cap_per_ep)), smooth,
                        color=color_list[i], linewidth=2.5, label=f'User {i}')
        ax.set_xlabel("Episode")
        ax.set_ylabel("User Capacity (bits/s/Hz)")
        ax.set_title("User Channel Capacity per Episode (Averaged)")
        ax.legend(loc='best', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/user_capacity.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


        ###############################
        # plot attacker capacity - per episode average
        ###############################
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(self.attacker_num):
            ax.plot(range(len(avg_att_cap_per_ep[:, i])), avg_att_cap_per_ep[:, i],
                    color=color_list[i + 2], alpha=0.35, linewidth=1.0)
            w = min(20, len(avg_att_cap_per_ep) // 3)
            if w >= 2 and len(avg_att_cap_per_ep) >= w:
                smooth = np.convolve(avg_att_cap_per_ep[:, i], np.ones(w)/w, mode='valid')
                ax.plot(range(w-1, len(avg_att_cap_per_ep)), smooth,
                        color=color_list[i + 2], linewidth=2.5, label=f'Attacker {i}')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Attacker Eavesdropping Capacity (bits/s/Hz)")
        ax.set_title("Attacker Eavesdropping Capacity per Episode (Averaged)")
        ax.legend(loc='best', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/attaker_capacity.png', dpi=200, bbox_inches='tight')
        plt.close('all')
        
        
        ###############################
        # plot ris
        ###############################
        for i in range(self.RIS_ant_num):
            self.plot_one_RIS_element(i)
            
        
        ###############################
        # plot trajectory
        ###############################
        self.plot_trajectory()

    
    def plot_one_RIS_element(self, index):
        """
        绘制单个RIS反射单元的状态 — 只画相位，按轮次平均
        """
        data = self.all_steps['RIS_elements'][index]
        if len(data) == 0:
            return

        # 计算相位
        phase_list = [cmath.phase(c) for c in data]

        # 按轮次平均（使用 step_num_per_episode）
        step_num_per_episode = []
        try:
            with open(self.store_path + 'step_num_per_episode.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    step_num_per_episode.append(int(row[0]))
        except:
            step_num_per_episode = [len(data)]

        # 按轮次求平均相位
        avg_phase_per_ep = []
        j = 0
        for steps in step_num_per_episode:
            if j + steps <= len(phase_list):
                avg_phase_per_ep.append(np.mean(phase_list[j:j+steps]))
            j += steps
        if not avg_phase_per_ep:
            avg_phase_per_ep = phase_list

        fig, ax = plt.subplots(figsize=(14, 5))
        episodes = range(len(avg_phase_per_ep))
        ax.plot(episodes, avg_phase_per_ep, color='#90CAF9', linewidth=1.2, alpha=0.6, label='Per-episode')
        # 滑动平均
        w = min(20, len(avg_phase_per_ep) // 3)
        if w >= 2 and len(avg_phase_per_ep) >= w:
            smooth = np.convolve(avg_phase_per_ep, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(avg_phase_per_ep)), smooth, color='#0D47A1', linewidth=2.5, label=f'{w}-ep Moving Avg')
        ax.set_ylim(-math.pi * 1.1, math.pi * 1.1)
        ax.set_yticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi])
        ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'], fontsize=14)
        ax.set_xlabel("Episode", fontsize=16)
        ax.set_ylabel("Phase (rad)", fontsize=16)
        ax.set_title(f"RIS Element #{index} — Reflection Phase", fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', labelsize=13)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/RIS/RIS_' + str(index) + '_element.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        
    def plot_trajectory(self):
        # get init location
        init_uav_coord = read_init_location(entity_type='UAV')
        init_user_coord_0 = read_init_location(entity_type='user', index=0)
        init_user_coord_1 = read_init_location(entity_type='user', index=1)
        init_attacker_coord = read_init_location(entity_type='attacker', index=0)

        ep_num = EP_NUM
        # 只显示 3 条轨迹：首、中、尾（更清晰）
        ep_list = [0, ep_num // 2, ep_num - 1]

        traj_colors = ['#1976D2', '#E64A19', '#388E3C']
        traj_labels = ['Episode 1 (Early)', f'Episode {ep_num//2+1} (Mid)', f'Episode {ep_num} (Final)']

        print(f"Trajectory episodes: {ep_list}")
        print(f"UAV init: {init_uav_coord}")

        fig, ax = plt.subplots(figsize=(14, 10))
        all_x, all_y = [], []

        for idx, i in enumerate(ep_list):
            filename = os.path.join(STORE_PATH, f'simulation_result_ep_{i}.mat')
            if not os.path.exists(filename):
                continue
            data = loadmat(filename)

            uav_coord_x = [init_uav_coord[0]]
            uav_coord_y = [init_uav_coord[1]]

            # store_list: [..., 'UAV_movement'(8), 'ARIS_amplification'(9)]
            # [-2] = UAV_movement, NOT [-1] which is ARIS_amplification
            result = data[f'result_{i}'][0][0]
            uav_movt = result[-2]  # UAV_movement (not ARIS_amplification)
            for j in range(uav_movt.shape[0]):
                uav_coord_x.append(uav_coord_x[-1] + uav_movt[j][0])
                uav_coord_y.append(uav_coord_y[-1] + uav_movt[j][1])

            all_x.extend(uav_coord_x)
            all_y.extend(uav_coord_y)

            color = traj_colors[idx % len(traj_colors)]
            # 用粗实线 + 箭头方向标记（每10步一个箭头）
            ax.plot(uav_coord_x, uav_coord_y,
                    color=color, linewidth=3.5, alpha=0.9,
                    solid_capstyle='round', label=traj_labels[idx])
            # 起点（大圆）
            ax.scatter(uav_coord_x[0], uav_coord_y[0],
                       color=color, marker='o', s=200, zorder=9,
                       edgecolors='white', linewidths=2)
            # 终点（大星）
            ax.scatter(uav_coord_x[-1], uav_coord_y[-1],
                       color=color, marker='*', s=500, zorder=9,
                       edgecolors='black', linewidths=0.5)
            # 每10步画一个小点表示方向
            for step in range(10, len(uav_coord_x), 10):
                ax.plot(uav_coord_x[step], uav_coord_y[step],
                        'o', color=color, markersize=6, alpha=0.7, zorder=7)

        # 标记固定实体（在最上层）
        # User 0
        ax.scatter(init_user_coord_0[0], init_user_coord_0[1],
                   c='#D32F2F', marker='s', s=400,
                   edgecolors='black', linewidths=2, zorder=11)
        ax.annotate('User 0', (init_user_coord_0[0], init_user_coord_0[1]),
                     textcoords="offset points", xytext=(15, -18),
                     fontsize=15, fontweight='bold', color='#B71C1C',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D32F2F', alpha=0.9))

        # User 1
        ax.scatter(init_user_coord_1[0], init_user_coord_1[1],
                   c='#1565C0', marker='s', s=400,
                   edgecolors='black', linewidths=2, zorder=11)
        ax.annotate('User 1', (init_user_coord_1[0], init_user_coord_1[1]),
                     textcoords="offset points", xytext=(15, -18),
                     fontsize=15, fontweight='bold', color='#0D47A1',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#1565C0', alpha=0.9))

        # RIS — actual position from init_location.xlsx (12, 35)
        ris_x, ris_y = 12, 35
        ax.scatter(ris_x, ris_y, c='#2E7D32', marker='^', s=500,
                   edgecolors='black', linewidths=2, zorder=11)
        ax.annotate('RIS', (ris_x, ris_y),
                     textcoords="offset points", xytext=(15, 10),
                     fontsize=15, fontweight='bold', color='#1B5E20',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2E7D32', alpha=0.9))

        # Eavesdropper (Attacker) — position from init_location.xlsx (-20, 10)
        attacker_x, attacker_y = init_attacker_coord[0], init_attacker_coord[1]
        ax.scatter(attacker_x, attacker_y, c='#6A1B9A', marker='X', s=500,
                   edgecolors='black', linewidths=2, zorder=11)
        ax.annotate('Eavesdropper', (attacker_x, attacker_y),
                     textcoords="offset points", xytext=(15, -18),
                     fontsize=15, fontweight='bold', color='#4A148C',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#6A1B9A', alpha=0.9))

        # 设置坐标范围 - 包含所有轨迹点和固定实体
        if all_x and all_y:
            margin = 5
            x_lo = min(min(all_x), init_user_coord_0[0], init_user_coord_1[0], ris_x, attacker_x) - margin
            x_hi = max(max(all_x), init_user_coord_0[0], init_user_coord_1[0], ris_x, attacker_x) + margin
            y_lo = min(min(all_y), init_user_coord_0[1], init_user_coord_1[1], ris_y, attacker_y) - margin
            y_hi = max(max(all_y), init_user_coord_0[1], init_user_coord_1[1], ris_y, attacker_y) + margin
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel('X Position (m)', fontsize=16)
        ax.set_ylabel('Y Position (m)', fontsize=16)
        ax.set_title('UAV Flight Trajectory', fontsize=20, fontweight='bold')
        ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(self.store_path + 'plot/trajectory.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


    def restruct(self):
        savemat(self.store_path + 'all_steps.mat',self.all_steps)
        return 0
if __name__ == '__main__':
    LoadPlotObject = LoadAndPlot(
        store_path = STORE_PATH,
        )
    LoadPlotObject.plot()
    LoadPlotObject.restruct()

    

