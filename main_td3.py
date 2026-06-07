
from env import MiniSystem
import time

import argparse
from env import *
from td3 import *
import matplotlib.pyplot as plt





'''Hyperparameter Setting'''

parser = argparse.ArgumentParser()
parser.add_argument('--dvc',type=str,default='cuda', help='running device: cuda or cpu')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--start_episode', type=int, default=int(500), help='Max training episode')
parser.add_argument('--Max_train_episode', type=int, default=int(12), help='Max training episode')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--update_every', type=int, default=100, help='Training Fraquency, in stpes')
parser.add_argument('--episodes', type=int, default=int(5000), help='Training episodes')
parser.add_argument('--T', type=int, default=int(200), help='Max step')

parser.add_argument('--RIS_ant_num',type=int,default=20,help='Number of RIS element')
parser.add_argument('--BS_ant_num',type=int,default=1,help='Number of BS antenna')
parser.add_argument('--BS_p_max',type=int,default=30,help='total power of BS antenna (dbm)')
parser.add_argument('--User_num',type=int,default=2,help='Number of user')
parser.add_argument('--Attacker_num',type=int,default=1,help='Number of attacker')
parser.add_argument('--P_J',type=int,default=-40, help='total power of RIS (dbm)')
parser.add_argument('--P_Beta',type=int, default=8,help='amplitude of RIS (db)')

parser.add_argument('--if_with_NOMA',type=bool,default=True,help='if_with_jamming')
parser.add_argument('--scheme',type=int, default=0,help='[0,1,2]=[MR,AR,AJ]')


parser.add_argument('--if_with_jamming',type=bool,default=True,help='if_with_jamming')
parser.add_argument('--if_with_reflect',type=bool,default=True,help='if_with_jamming')
parser.add_argument('--Qos_constrain',type=int,default=1,help='Qos_constrain of users')
parser.add_argument('--Pr',type=int,default=0,help='Qos_constrain of RIS power per step (dbm)')
parser.add_argument('--c_error',type=float,default=0,help='imperfect CSI ')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hidden_size', type=int, default=[512,256], help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.dvc = device
if args.dvc.type == 'cuda':
    print("Device is CUDA")
else:
    print("Device is CPU")

# 0 是MR, 1 是AR， 2是 AJ
if 0 ==args.scheme:
    args.if_with_jamming = True
    args.if_with_reflect = True
elif 1 ==args.scheme:
    args.if_with_jamming = False
    args.if_with_reflect = True
elif 2 == args.scheme:
    args.if_with_jamming = True
    args.if_with_reflect = False


if __name__ == '__main__':
    # --------------------------↓↓↓↓↓ save path ↓↓↓↓↓----------------------#
    # 指定你要清空的文件夹路径
    directory_path = './data/TD3-train'
    ensure_directory_exists(directory_path)
    # 调用函数
    clear_folder(directory_path)
    save_model_path = './models/TD3'
    ensure_directory_exists(save_model_path)


    # --------------------------↓↓↓↓↓ train ↓↓↓↓↓----------------                                                                      ------#
    RIS = [20,30]
    beta = [10]
    CSI = [0,0.2,0.4,0.6,0.8]

    for r in RIS:
        # args.P_Beta = b
        args.RIS_ant_num = r

        np.random.seed(0)
        torch.manual_seed(0)

        system = MiniSystem(**vars(args))
        agent = TD3(env=system, **vars(args))

        max_action = np.ones(system.action_dim)

        expl_noise = 0.4
        print(args)

        episode_rewards = []
        episode_secrecys = []

        save_parameter = f'M{args.RIS_ant_num}_b{args.P_Beta}_R{args.if_with_reflect}_J{args.if_with_jamming}'
        ensure_directory_exists('./data/TD3_evaluate/e_r_s')
        with open(f'./data/TD3_evaluate/e_r_s/{save_parameter}.txt', 'w') as f:
            for episode in range(args.episodes):
                start_time = time.time()  # 记录开始时间
                system.reset()
                state = system.observe()
                episode_reward = 0
                episode_secrecy = 0
                # if episode == 200:
                #     expl_noise = 0.2
                # elif episode>200:
                expl_noise *= 0.999


                episode_SEE = 0
                episode_EC = 0
                capacity = np.zeros((system.user_num, 1), dtype=float)
                attacker_rate = np.zeros((system.user_num, 1), dtype=float)

                uav_trajectory = []
                uav_coordinate = system.UavRis.coordinate.copy()
                uav_trajectory.append(uav_coordinate)


                for t in range(args.T):
                    # get action by using state
                    if t == args.T-1:
                        done = True
                    else:
                        done = False

                    action = agent.select_action(state)
                    action = (action + np.random.normal(0, max_action * expl_noise, size=system.action_dim)).clip(
                        -max_action, max_action)

                    next_state, reward = system.step(action)
                    episode_reward += reward
                    episode_SEE += system.SEE
                    episode_EC += system.UavRis.fly_energy


                    agent.replay_buffer.add(state,action,reward, next_state, done)
                    state = next_state


                    uav_coordinate = system.UavRis.coordinate.copy()
                    uav_trajectory.append(uav_coordinate)


                    for user in system.user_list:
                        episode_secrecy += user.secrecy
                        capacity[user.index] += user.capacity
                        attacker_rate[user.index] += max(user.attack_rate)

                    if episode > args.Max_train_episode and t%2==0:
                        agent.Update(args.batch_size)
                capacity = capacity / args.T
                attacker_rate = attacker_rate / args.T

                episode_rewards.append(episode_reward)
                episode_secrecys.append(episode_secrecy)


                if episode % 20 ==0:

                    uav_trajectory = np.array(uav_trajectory)
                    uav_x = uav_trajectory[:, 0]
                    uav_y = uav_trajectory[:, 1]
                    savepath = f'{directory_path}/traj_{save_parameter}.png'
                    draw_location(uav_x, uav_y,system, args.T, episode, savepath=savepath)

                if episode == (args.episodes - 1) or (episode % 500==0 and expl_noise < 0.1):
                    agent.save(save_model_path,
                               args.RIS_ant_num,
                               args.P_Beta,
                               args.BS_p_max,
                               args.BS_ant_num,
                               args.c_error,
                               args.if_with_jamming,
                               args.P_J)
                    print("save models success")
                end_time = time.time()  # 记录结束时间
                skd = sum(system.UavRis.skds) / args.T / args.RIS_ant_num
                f.write(f'Episode: {episode}, reward: {episode_reward}, secrecy: {episode_secrecy}, SEE: {episode_SEE / args.T}\n')
                print(f"RIS:{args.RIS_ant_num}_epi:{episode}, reward:{episode_reward:.3f},"
                      f"ave_sec:{episode_secrecy/args.T:.3f},cap0:{capacity[0]},"
                      f"att0:{attacker_rate[0]},cap1: {capacity[1]},att1:{attacker_rate[1]},"
                      f"SEE:{episode_SEE / args.T:.3f},noise:{expl_noise:.3f},skd:{skd:.3f} "
                      f"时间: {(end_time - start_time):.3f}秒")

        with open(f'./data/TD3_evaluate/e_r_s/{save_parameter}.txt', 'r') as f:
            episodes = []
            rewards = []
            secrecys = []
            SEE = []
            for line in f:
                # 解析每一行，提取 episode和 sum_see
                parts = line.split(', ')
                episodes.append(int(parts[0].split(': ')[1]))
                rewards.append(float(parts[1].split(': ')[1]))
                secrecys.append(float(parts[2].split(': ')[1]))
                SEE.append(float(parts[3].split(': ')[1].strip('[]\n')))  # 删除额外字符

        for data, name in [(rewards, 'reward'), (secrecys, 'secrecys'), (SEE, 'SEE')]:
            plt.plot(episodes, data)
            plt.xlabel('Episodes')
            plt.ylabel(f'{name}')
            plt.title('TD3 ')
            directory_path = './data/TD3-train'
            ensure_directory_exists(directory_path)
            path = directory_path + f'/{name}_{save_parameter}.png'
            plt.savefig(path)
            plt.close()



