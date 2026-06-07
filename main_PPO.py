
import argparse
from PPO import *
from utils import *
from env import MiniSystem
import matplotlib.pyplot as plt



'''Hyperparameter Setting'''

parser = argparse.ArgumentParser()
parser.add_argument('--dvc',type=str,default='cuda', help='running device: cuda or cpu')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_episode', type=int, default=int(400), help='Max training episode')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
parser.add_argument('--episodes', type=int, default=int(5000), help='Training episodes')
parser.add_argument('--T', type=int, default=int(200), help='Max step')

parser.add_argument('--RIS_ant_num',type=int,default=20,help='Number of RIS element')
parser.add_argument('--BS_ant_num',type=int,default=1,help='Number of BS antenna')
parser.add_argument('--BS_p_max',type=int,default=30,help='total power of BS antenna (dbm)')
parser.add_argument('--User_num',type=int,default=2,help='Number of user')
parser.add_argument('--Attacker_num',type=int,default=1,help='Number of attacker')
parser.add_argument('--P_J',type=int,default=-40, help='total power of RIS (dbm)')
parser.add_argument('--P_Beta',type=int, default=8,help='amplitude of RIS (db)')
parser.add_argument('--scheme',type=int, default=0,help='[0,1,2]=[MR,AR,AJ]')

parser.add_argument('--if_with_jamming',type=bool,default=True,help='if_with_jamming')
parser.add_argument('--if_with_reflect',type=bool,default=True,help='if_with_jamming')
parser.add_argument('--Qos_constrain',type=int,default=1,help='Qos_constrain of users')
parser.add_argument('--Pr',type=int,default=0,help='Qos_constrain of RIS power per step (dbm)')
parser.add_argument('--c_error',type=float,default=0,help='imperfect CSI ')


parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=int, default=0.95, help='clamp')
parser.add_argument('--hidden_size', type=int, default=[512,256], help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--epochs', type=int, default=20, help='train frequency')
parser.add_argument('--eps', type=int, default=0.2, help='clamp')

parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')


args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# args.dvc = device
args.dvc = torch.device(args.dvc) # from str to torch.device
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



print("Random Seed: {}".format(args.seed))

print(args)

RIS = [20]
beta = [4]

def main():
    # 指定你要清空的文件夹路径
    directory_path = './data/PPO-train'
    ensure_directory_exists(directory_path)
    ensure_directory_exists('./data/PPO_evaluate/e_r_s')
    # 调用函数
    clear_folder(directory_path)

    for r in RIS:
        args.RIS_ant_num = r
        # args.P_Beta = b
        # Seed Everything
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        save_models_path = './models/PPO'
        ensure_directory_exists(save_models_path)

        system = MiniSystem(**vars(args))
        kwargs = {
            "state_dim": system.state_dim,
            "action_dim": system.action_dim,
            "env_with_Dead": True,
            "gamma": 0.94,
            "net_width": 256,
            "a_lr": 3e-4,
            "c_lr": 3e-4,
        }
        agent = PPO(**kwargs)


        episode_rewards = []
        episode_secrecys = []
        EC = []

        save_parameter = f'N{args.BS_ant_num}_M{args.RIS_ant_num}_b{args.P_Beta}_P{args.BS_p_max}_e{args.c_error}_J{args.if_with_jamming}'
        with open(f'./data/PPO_evaluate/e_r_s/{save_parameter}.txt', 'w') as f:
            for episode in range(args.episodes):

                system.reset()
                s = system.observe()


                episode_reward = 0
                episode_secrecy = 0
                episode_P_RIS = 0
                episode_SEE = 0
                episode_EC = 0
                capacity = np.zeros((system.user_num, 1), dtype=float)
                attacker_rate = np.zeros((system.user_num, 1), dtype=float)

                uav_trajectory = []
                uav_coordinate = system.UavRis.coordinate.copy()
                uav_trajectory.append(uav_coordinate)

                for step in range(args.T):
                    a, logprob_a = agent.select_action(s) #a_low=[0,1]
                    action = a*2-1

                    s_next, r = system.step(action)
                    done = False
                    dw = False
                    episode_reward += r
                    for user in system.user_list:
                        episode_secrecy += user.secrecy
                        capacity[user.index] += user.capacity
                        attacker_rate[user.index] += max(user.attack_rate)

                    episode_P_RIS += system.P_RIS
                    episode_SEE += system.SEE
                    episode_EC += system.UavRis.fly_energy / system.UavRis.step_time
                    EC.append(system.UavRis.vel)

                    agent.put_data((s, a, r, s_next, logprob_a, done, dw))

                    s= s_next

                    uav_coordinate = system.UavRis.coordinate.copy()
                    uav_trajectory.append(uav_coordinate)


                if episode % 2 == 0:
                    # for step in range(args.upgrade_step):
                    agent.train()
                if episode % 20 == 0:
                    uav_trajectory = np.array(uav_trajectory)
                    uav_x = uav_trajectory[:, 0]
                    uav_y = uav_trajectory[:, 1]
                    savepath = f'{directory_path}/traj_{save_parameter}.png'
                    draw_location(uav_x, uav_y, system, args.T, args.if_with_jamming, savepath=savepath)
                if episode == (args.episodes - 1) or (episode % 500 == 0 and episode > 2000):
                    agent.save(save_models_path,
                               args.RIS_ant_num,
                               args.P_Beta,
                               args.if_with_reflect,
                               args.if_with_jamming
                               )
                    print("save agent success")
                capacity = capacity / args.T
                attacker_rate = attacker_rate / args.T
                skd = sum(system.UavRis.skds) / args.T / args.RIS_ant_num


                f.write(f'Episode: {episode}, reward: {episode_reward}, secrecy: {episode_secrecy}, SEE: {episode_SEE / args.T}\n')
                print(f"RIS:{args.RIS_ant_num}episode:{episode}, reward:{episode_reward:.3f},"
                      f"secrecy:{episode_secrecy / args.T:.3f},A_SEE:{episode_SEE / args.T:.3f},"
                      f"cap0: {capacity[0]},att0:{attacker_rate[0]},cap1:{capacity[1]},"
                      f"att1:{attacker_rate[1]},skd:{skd:.3f},"
                      f"时间: 秒")

        rewards = []
        secrecys = []
        episodes = []
        SEE = []

        with open(f'./data/PPO_evaluate/e_r_s/{save_parameter}.txt', 'r') as f:
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
            plt.title('PPO ')
            directory_path = './data/PPO-train'
            ensure_directory_exists(directory_path)
            path = directory_path + f'/{name}_{save_parameter}.png'
            plt.savefig(path)
            plt.close()


if __name__ == '__main__':
    main()
