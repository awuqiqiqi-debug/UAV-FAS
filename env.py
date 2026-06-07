#%matplotlib inline
import numpy as np
from entity import *
from channel import *
from math_tool import *
from datetime import datetime
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from render import Render
from data_manager import DataManager
# s.t every simulition is the same model
np.random.seed(2)

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
delta_time = 0.1/1000 #0.1ms

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

ENERGY_IDLE = get_energy_consumption(0)
ENERGY_MAX_MOVEMENT = get_energy_consumption(0.5)

######################################################


class MiniSystem(object):
#class MiniSystem(K=1):
    """
    define mini RIS communication system with one UAV
        and one RIS and one user, one attacker
    """
    def __init__(self, UAV_num = 1, RIS_num = 1, user_num = 1, attacker_num = 1, fre = 28e9, \
                 RIS_ant_num = 16, UAV_ant_num=8, if_dir_link = 1, if_with_RIS = True, \
                 if_move_users = True, if_movements = True, reverse_x_y = (True, True), \
                 if_UAV_pos_state = True, reward_design = 'ssr', project_name = None, step_num=100):
        self.if_dir_link = if_dir_link
        self.if_with_RIS = if_with_RIS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.border = [(-25,25), (0, 50)]
        # 1.init entities: 1 UAV, 1 RIS, many users and attackers
        self.data_manager = DataManager(file_path='./data', project_name = project_name, \
        store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity', 'secure_capacity', 'attaker_capacity','G_power', 'reward','UAV_movement', 'ARIS_amplification', 'FAS_port', 'RIS_scheduling'])
        # 1.1 init UAV position and beamforming matrix (UAV_BS_FAS: 基站+波束可重构离散端口流体天线)
        self.UAV = UAV_BS_FAS(
            coordinate=self.data_manager.read_init_location('UAV', 0),
            bs_ant_num=UAV_ant_num, bs_ant_type='ULA',
            fas_num_ports=8, fas_ant_type='UPA',
            max_movement_per_time_slot=0.5)
        self.UAV.G = np.asmatrix(np.ones((self.UAV.bs_ant_num, user_num), dtype=complex), dtype=complex)
        self.UAV.F = np.asmatrix(np.ones((self.UAV.fas_num_ports, user_num), dtype=complex), dtype=complex)
        self.power_factor = 100
        self.UAV.G_Pmax = np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        self.UAV.F_Pmax = np.trace(self.UAV.F * self.UAV.F.H) * self.power_factor
        # 1.2 init RIS (多功能RIS: 反射+干扰)
        self.RIS = RIS(\
        coordinate=self.data_manager.read_init_location('RIS', 0), \
        coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        ant_num=RIS_ant_num, ant_type='UPA',
        Pr=30, P_J=30, beta=20, sigma=-100)
        # 1.3 init users
        self.user_list = []
        
        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = -114
            self.user_list.append(user)

        # 1.4 init attackers
        self.attacker_list = []
        
        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros((user_num))
            attacker.noise_power = -114
            self.attacker_list.append(attacker)
        # 1.5 generate the eavesdrop capacity array , shape: P X K
        self.eavesdrop_capacity_array= np.zeros((attacker_num, user_num))
        
        # 1.6 reward design
        self.reward_design = reward_design # reward_design is ['ssr' or 'see']

        # 1.7 step_num
        self.step_num = step_num
        
        # 2.init channel (UAV/BS/FAS → RIS → 用户/窃听者)
        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)  # BS→RIS
        self.h_U_k = []  # BS → 用户 (直传，保留用于信道对齐)
        self.h_R_k = []  # RIS → 用户
        self.h_U_p = []  # BS → 攻击者 (直传，保留用于信道对齐)
        self.h_R_p = []  # RIS → 攻击者
        self.h_F_k = []  # FAS → 用户 (直传，保留用于信道对齐)
        self.h_F_p = []  # FAS → 攻击者 (直传，保留用于信道对齐)

        # 创建FAS实体用于信道建模（使用fas_num_ports作为天线数）
        from entity import User as UserEntity
        self.FAS_entity = UserEntity(
            coordinate=self.UAV.coordinate.copy(), index=0,
            ant_num=self.UAV.fas_num_ports, ant_type='UPA')
        self.FAS_entity.type = 'UAV'  # 伪装为UAV类型以匹配信道模型

        # h_FR: FAS→RIS信道 (FAS信号通过RIS反射)
        self.h_FR = mmWave_channel(self.FAS_entity, self.RIS, fre)

        for user_k in self.user_list:
            self.h_U_k.append(mmWave_channel(user_k, self.UAV, fre))
            self.h_R_k.append(mmWave_channel(user_k, self.RIS, fre))
            self.h_F_k.append(mmWave_channel(user_k, self.FAS_entity, fre))
        for attacker_p in self.attacker_list:
            self.h_U_p.append(mmWave_channel(attacker_p, self.UAV, fre))
            self.h_R_p.append(mmWave_channel(attacker_p, self.RIS, fre))
            self.h_F_p.append(mmWave_channel(attacker_p, self.FAS_entity, fre))

        # 3 update user and attaker channel capacity
        self.update_channel_capacity()

        # 4 draw system
        self.render_obj = Render(self)      
        
    def reset(self):
        """
        reset UAV, users, attackers, beamforming matrix, reflecting coefficient
        """
        # 1 reset UAV
        self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))
        # Reset FAS entity coordinate
        self.FAS_entity.coordinate = self.UAV.coordinate.copy()
        # 2 reset users
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)
        # 3 reset attackers
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)
        # 4 reset beamforming matrix (BS + FAS)
        self.UAV.G = np.asmatrix(np.ones((self.UAV.bs_ant_num, self.user_num), dtype=complex), dtype=complex)
        self.UAV.G_Pmax = np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        self.UAV.F = np.asmatrix(np.ones((self.UAV.fas_num_ports, self.user_num), dtype=complex), dtype=complex)
        self.UAV.F_Pmax = np.trace(self.UAV.F * self.UAV.F.H) * self.power_factor
        self.UAV.fas_active_port = 0
        # 5 reset RIS phase matrices (多功能RIS: 反射+干扰)
        self.RIS.theta_R = np.asmatrix(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype=complex)
        self.RIS.theta_J = np.asmatrix(np.diag(np.zeros(self.RIS.ant_num, dtype=complex)), dtype=complex)
        self.RIS.theta = np.asmatrix(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype=complex)
        self.RIS.Phi = np.asmatrix(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype=complex)
        self.RIS.skd_rate = 1
        self.RIS.P_RIS = []
        self.RIS.betas = []
        self.RIS.skds = []
        # 6 reset time
        self.render_obj.t_index = 0
        # 7 reset CSI
        self.H_UR.update_CSI()
        self.h_FR.update_CSI()
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p + self.h_F_k + self.h_F_p:
            h.update_CSI()
        # 8 reset capcaity
        self.update_channel_capacity()

    def step(self, action_0 = 0, action_1 = 0, G = 0, F = 0, Phi = 0, set_pos_x = 0, set_pos_y = 0):
        """
        test step only move UAV and update channel
        """
        # 0 update render
        
        self.render_obj.t_index += 1
        # 1 update entities
        
        if self.if_move_users:
            self.user_list[0].update_coordinate(0.2, -1/2 * math.pi)
            self.user_list[1].update_coordinate(0.2, -1/2 * math.pi)

        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot

            ######################################################
            # new for energy
            v_t = (move_x ** 2 + move_y ** 2) ** 0.5
            ######################################################

            if self.reverse_x_y[0]:
                move_x = -move_x

            if self.reverse_x_y[1]:
                move_y = -move_y

            self.UAV.coordinate[0] +=move_x
            self.UAV.coordinate[1] +=move_y
            self.FAS_entity.coordinate = self.UAV.coordinate.copy()  # 同步FAS坐标
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV.coordinate[0] = set_pos_x
            self.UAV.coordinate[1] = set_pos_y
            self.FAS_entity.coordinate = self.UAV.coordinate.copy()  # 同步FAS坐标

        # 2 update channel CSI

        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p + self.h_F_k + self.h_F_p:
            h.update_CSI()
        # !!! test to make direct link zero
        if self.if_dir_link == 0:
            for h in self.h_U_k + self.h_U_p + self.h_F_k + self.h_F_p:
                h.channel_matrix = np.asmatrix(np.zeros(shape = np.shape(h.channel_matrix)), dtype=complex)
        if self.if_with_RIS == False:
            self.H_UR.channel_matrix = np.asmatrix(np.zeros((self.RIS.ant_num, self.UAV.bs_ant_num)), dtype=complex)
            self.h_FR.channel_matrix = np.asmatrix(np.zeros((self.RIS.ant_num, self.UAV.fas_ant_num)), dtype=complex)
        else:
            self.H_UR.update_CSI()
            self.h_FR.update_CSI()
        # 3 update beamforming matrix (BS + FAS) & RIS phase matrices
        self.UAV.G = convert_list_to_complex_matrix(G, (self.UAV.bs_ant_num, self.user_num)) * math.pow(self.power_factor, 0.5)
        self.UAV.F = convert_list_to_complex_matrix(F, (self.UAV.fas_num_ports, self.user_num)) * math.pow(self.power_factor, 0.5)
        # 归一化G和F，确保功率不超过Pmax
        P_G = np.trace(self.UAV.G * self.UAV.G.H).real
        if P_G > self.UAV.G_Pmax:
            self.UAV.G *= math.sqrt(self.UAV.G_Pmax / P_G)
        P_F = np.trace(self.UAV.F * self.UAV.F.H).real
        if P_F > self.UAV.F_Pmax:
            self.UAV.F *= math.sqrt(self.UAV.F_Pmax / P_F)

        # FAS端口选择：自动优化选择最优端口
        if len(self.attacker_list) > 0:
            best_port, _, _ = self.UAV.fas_beam_optimize(
                self.RIS.coordinate, self.attacker_list[0].coordinate)
            self.UAV.select_port(best_port)

        # 多功能RIS：更新相位矩阵（反射+干扰）
        if self.if_with_RIS:
            # 计算用户和攻击者的信道对齐相位
            UA_theta = []
            for user in self.user_list:
                theta_user = self.RIS.calculate_channel_alignment_theta(
                    self.H_UR.channel_matrix, self.h_R_k[user.index].channel_matrix,
                    self.h_U_k[user.index].channel_matrix)
                UA_theta.append(theta_user)
            for attacker in self.attacker_list:
                theta_attacker = self.RIS.calculate_channel_alignment_theta(
                    self.H_UR.channel_matrix, self.h_R_p[attacker.index].channel_matrix,
                    self.h_U_p[attacker.index].channel_matrix)
                UA_theta.append(theta_attacker)

            # ====== 启发式相位优化（v5: 全部元素用于反射+干扰，无需RL调度） ======
            h_rk_list = [self.h_R_k[u.index].channel_matrix for u in self.user_list]
            h_bk_list = [self.h_U_k[u.index].channel_matrix for u in self.user_list]
            h_rp_list = [self.h_R_p[a.index].channel_matrix for a in self.attacker_list]

            # 模块1：联合反射相位（增强用户、抵消窃听者）
            heuristic_reflect, heuristic_jam_base = self.RIS.calculate_joint_phase_optimization(
                self.H_UR.channel_matrix, h_rk_list, h_bk_list, h_rp_list, alpha_cancel=0.0)

            # 模块2：自适应干扰功率
            P_J_base = self.RIS._dbm_to_mw(self.RIS.P_J)
            P_J_max = P_J_base * 5
            P_J_adaptive = self.RIS.calculate_adaptive_jamming_power(
                h_rp_list[0], h_rk_list, P_J_base, P_J_max)

            # 模块2：定向干扰波束（主瓣→窃听者，不做用户零陷）
            heuristic_jam = self.RIS.calculate_jamming_beamforming(
                self.H_UR.channel_matrix, h_rk_list, h_rp_list)
            # ====== 启发式优化结束 ======

            # v5: Phi由RL控制已移除，全部RIS相位由启发式确定
            # 传入dummy action给update_phase_matrices（仅用于beta提取，beta会被heuristic覆盖）
            Phi_dummy = np.ones(self.RIS.ant_num)
            Phi_full = np.concatenate([Phi_dummy, np.zeros(self.RIS.ant_num)])
            self.RIS.update_phase_matrices(
                Phi_full, UA_theta,
                if_with_jamming=True, if_with_reflect=True,
                heuristic_reflect=heuristic_reflect,
                heuristic_jam=heuristic_jam)

            # 限制RIS功率
            self.RIS.limit_power(self.H_UR.channel_matrix, self.UAV.G)
        # 4 update channel capacity in every user and attacker
        self.update_channel_capacity()
        # 5 store current system state to .mat
        self.store_current_system_sate()
        # 6 get new state
        new_state = self.observe()
        # 7 get reward
        reward = self.reward()
        
        # 7.1 SEE: 保密速率/能量 (暂不惩罚能量，让SSR主导学习)
        ######################################################
        ######################################################

        # 8 calculate if UAV is cross the bourder
        done = False
        x, y = self.UAV.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
            reward = -10
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -10
        self.data_manager.store_data([reward],'reward')
        return new_state, reward, done, []

    def reward(self):
        """
        奖励函数：SSR + RIS接近度 + 窃听者抑制 + 干扰效果 + 用户零陷验证
        """
        reward = 0
        P = np.trace(self.UAV.G * self.UAV.G.H)

        if abs(P) > abs(self.UAV.G_Pmax):
            reward = abs(self.UAV.G_Pmax) - abs(P)
            reward /= self.power_factor
        else:
            for user in self.user_list:
                r = user.capacity - max(self.eavesdrop_capacity_array[:, user.index])
                reward += r / (self.user_num * 2)

        # 窃听者惩罚（加倍）
        for attacker in self.attacker_list:
            eavesdrop_penalty = -0.02 * np.mean(attacker.capacity)
            reward += eavesdrop_penalty

        # 保密速率奖励
        total_secrecy = sum(user.secure_capacity for user in self.user_list)
        secrecy_bonus = 0.3 * min(1.0, total_secrecy / self.user_num)
        reward += secrecy_bonus

        # RIS干扰效果奖励 + 用户零陷验证
        if self.if_with_RIS and len(self.attacker_list) > 0:
            jamming_effect = 0
            user_jamming_harm = 0
            for attacker in self.attacker_list:
                jamming_power = self.RIS.calculate_jamming_power(
                    self.H_UR.channel_matrix, self.h_R_p[attacker.index].channel_matrix,
                    self.UAV.G, self.h_FR.channel_matrix, self.UAV.F)
                jamming_effect += jamming_power
            # 检查干扰是否泄漏到用户
            for user in self.user_list:
                h_R_user = self.h_R_k[user.index].channel_matrix
                user_jam_from_ris = math.pow(np.linalg.norm(h_R_user.H * self.RIS.theta_J), 2)
                user_jamming_harm += user_jam_from_ris
            # 奖励窃听者干扰，惩罚用户泄漏
            jam_bonus = 0.5 * min(1.0, jamming_effect / 5.0)
            jam_penalty = -0.3 * min(1.0, user_jamming_harm / 0.1)
            reward += jam_bonus + jam_penalty

        # RIS接近度奖励
        ris_dist = ((self.UAV.coordinate[0] - self.RIS.coordinate[0])**2 +
                    (self.UAV.coordinate[1] - self.RIS.coordinate[1])**2) ** 0.5
        max_dist = 50
        ris_bonus = 0.5 * (1 - ris_dist / max_dist)
        reward += ris_bonus

        return reward
    
    def observe(self):
        """
        used in function main to get current state
        the state is a list with
        """
        # users' and attackers' comprehensive channel
        comprehensive_channel_elements_list = []
        for entity in self.user_list + self.attacker_list:
            tmp_list = list(np.array(np.reshape(entity.comprehensive_channel, (1,-1)))[0])
            comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list))
        UAV_position_list = []
        if self.if_UAV_pos_state:
            UAV_position_list = list(self.UAV.coordinate)

        # FAS端口状态：当前激活端口索引（归一化到[-1,1]）
        fas_port_state = [(self.UAV.fas_active_port / max(1, self.UAV.fas_num_ports - 1)) * 2 - 1]

        # RIS调度状态：反射单元占比
        ris_scheduling_state = [self.RIS.skd_rate * 2 - 1]  # 归一化到[-1,1]

        return comprehensive_channel_elements_list + UAV_position_list + fas_port_state + ris_scheduling_state

    def store_current_system_sate(self):
        """
        function used in step() to store system state
        """
        # 1 store beamforming matrix
        row_data = list(np.array(np.reshape(self.UAV.G, (1, -1)))[0,:])
        self.data_manager.store_data(row_data, 'beamforming_matrix')
        # 2 store reflecting coefficient matrix
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1,-1)))[0,:])      
        self.data_manager.store_data(row_data, 'reflecting_coefficient')
        # 3 store UAV state
        row_data = list(self.UAV.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')
        # 4 store user_capicity
        row_data = [user.secure_capacity for user in self.user_list] \
        + [user.capacity for user in self.user_list]
        # 5 store G_power (BS + FAS)
        row_data = [np.trace(self.UAV.G*self.UAV.G.H), self.UAV.G_Pmax,
                    np.trace(self.UAV.F*self.UAV.F.H), self.UAV.F_Pmax]
        self.data_manager.store_data(row_data, 'G_power')
        row_data = []
        for user in self.user_list:
            row_data.append(user.capacity)
        self.data_manager.store_data(row_data, 'user_capacity')

        row_data = []
        for attacker in self.attacker_list:
            row_data.append(attacker.capacity)
        self.data_manager.store_data(row_data, 'attaker_capacity')

        row_data = []
        for user in self.user_list:
            row_data.append(user.secure_capacity)
        self.data_manager.store_data(row_data, 'secure_capacity')

        # 6 store ARIS amplification factors
        ris_amplification = np.abs(np.diag(self.RIS.Phi))
        self.data_manager.store_data(list(ris_amplification), 'ARIS_amplification')

        # 7 store FAS port state and RIS scheduling
        fas_port_state = [self.UAV.fas_active_port]
        self.data_manager.store_data(fas_port_state, 'FAS_port')

        ris_scheduling = [self.RIS.skd_rate]
        self.data_manager.store_data(ris_scheduling, 'RIS_scheduling')


    def update_channel_capacity(self):
        """
        function used in step to calculate user and attackers' capacity
        综合信道 = BS综合信道 + FAS综合信道 (垂直拼接)
        """
        # 1 calculate eavesdrop rate
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
            h_BS = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
            h_FAS = self.calculate_comprehensive_fas_channel_of_attacker_p(attacker.index)
            attacker.comprehensive_channel = np.hstack((h_BS, h_FAS))
        # 2 calculate unsecure rate
        for user in self.user_list:
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            h_BS = self.calculate_comprehensive_channel_of_user_k(user.index)
            h_FAS = self.calculate_comprehensive_fas_channel_of_user_k(user.index)
            user.comprehensive_channel = np.hstack((h_BS, h_FAS))

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """
        BS综合信道 for attacker p
        BS直传 + BS→RIS反射路径
        """
        h_U_p = self.h_U_p[p].channel_matrix  # BS→attacker, shape (4, 1)
        h_R_p = self.h_R_p[p].channel_matrix  # RIS→attacker, shape (4, 1)
        # BS直传 + RIS反射: h_U_p^H + h_R_p^H * theta_R * H_UR
        return h_U_p.H + h_R_p.H * self.RIS.theta_R * self.H_UR.channel_matrix

    def calculate_comprehensive_fas_channel_of_attacker_p(self, p):
        """
        FAS综合信道 for attacker p
        FAS直传 + FAS→RIS反射路径
        """
        h_F_p = self.h_F_p[p].channel_matrix  # FAS→attacker, shape (8, 1)
        h_R_p = self.h_R_p[p].channel_matrix  # RIS→attacker, shape (4, 1)
        # FAS直传 + RIS反射: h_F_p^H + h_R_p^H * theta_R * h_FR
        return h_F_p.H + h_R_p.H * self.RIS.theta_R * self.h_FR.channel_matrix

    def calculate_comprehensive_channel_of_user_k(self, k):
        """
        BS综合信道 for user k
        BS直传 + BS→RIS反射路径
        """
        h_U_k = self.h_U_k[k].channel_matrix  # BS→user, shape (4, 1)
        h_R_k = self.h_R_k[k].channel_matrix  # RIS→user, shape (4, 1)
        # BS直传 + RIS反射: h_U_k^H + h_R_k^H * theta_R * H_UR
        return h_U_k.H + h_R_k.H * self.RIS.theta_R * self.H_UR.channel_matrix

    def calculate_comprehensive_fas_channel_of_user_k(self, k):
        """
        FAS综合信道 for user k
        FAS直传 + FAS→RIS反射路径
        """
        h_F_k = self.h_F_k[k].channel_matrix  # FAS→user, shape (8, 1)
        h_R_k = self.h_R_k[k].channel_matrix  # RIS→user, shape (4, 1)
        # FAS直传 + RIS反射: h_F_k^H + h_R_k^H * theta_R * h_FR
        return h_F_k.H + h_R_k.H * self.RIS.theta_R * self.h_FR.channel_matrix

    def calculate_capacity_of_user_k(self, k):
        """
        function used in update_channel_capacity to calculate one user
        BS和FAS两路信号联合计算容量
        """
        noise_power = self.user_list[k].noise_power
        # BS综合信道 (bs_ant_num × 1)
        h_BS = self.calculate_comprehensive_channel_of_user_k(k)
        # FAS综合信道 (fas_ant_num × 1)
        h_FAS = self.calculate_comprehensive_fas_channel_of_user_k(k)
        G_k = self.UAV.G[:, k]  # [bs_ant_num, 1]
        F_k = self.UAV.F[:, k]  # [fas_ant_num, 1]
        # 期望信号: BS路径 + FAS路径
        alpha_k = math.pow(abs(h_BS * G_k + h_FAS * F_k), 2)
        # 干扰信号: 其他用户的BS+FAS波束成形
        if len(self.user_list) == 1:
            interference = np.asmatrix(np.zeros((1, 1), dtype=complex), dtype=complex)
        else:
            G_k_ = np.hstack((self.UAV.G[:, 0:k], self.UAV.G[:, k+1:]))
            F_k_ = np.hstack((self.UAV.F[:, 0:k], self.UAV.F[:, k+1:]))
            interference = h_BS * G_k_ + h_FAS * F_k_
        beta_k = math.pow(np.linalg.norm(interference), 2) + dB_to_normal(noise_power) * 1e-3
        return math.log10(1 + abs(alpha_k / beta_k))

    def calculate_capacity_array_of_attacker_p(self, p):
        """
        function used in update_channel_capacity to calculate one attacker capacities to K users
        BS和FAS两路信号联合计算容量，包含RIS干扰作为额外噪声
        output is a K length np.array, shape: (K,)
        """
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        h_BS = self.calculate_comprehensive_channel_of_attacker_p(p)
        h_FAS = self.calculate_comprehensive_fas_channel_of_attacker_p(p)
        h_R_p = self.h_R_p[p].channel_matrix  # RIS→attacker

        # 计算RIS干扰功率（放大热噪声模型）
        jamming_power = 0
        if self.if_with_RIS:
            # theta_J放大热噪声产生人工噪声，独立于数据信号
            # 干扰功率 = ||h_R_p^H * theta_J||^2 * P_J_mw
            P_J_mw = self.RIS._dbm_to_mw(self.RIS.P_J)
            jamming_power = math.pow(np.linalg.norm(h_R_p.H * self.RIS.theta_J), 2) * P_J_mw

        if K == 1:
            alpha_p = math.pow(abs(h_BS * self.UAV.G + h_FAS * self.UAV.F), 2)
            beta_p = jamming_power + dB_to_normal(noise_power) * 1e-3
            return np.array([math.log10(1 + abs(alpha_p / beta_p))])
        else:
            result = np.zeros(K)
            for k in range(K):
                G_k = self.UAV.G[:, k]
                F_k = self.UAV.F[:, k]
                alpha_p = math.pow(abs(h_BS * G_k + h_FAS * F_k), 2)
                G_k_ = np.hstack((self.UAV.G[:, 0:k], self.UAV.G[:, k+1:]))
                F_k_ = np.hstack((self.UAV.F[:, 0:k], self.UAV.F[:, k+1:]))
                interference = h_BS * G_k_ + h_FAS * F_k_
                beta_p = math.pow(np.linalg.norm(interference), 2) + jamming_power + dB_to_normal(noise_power) * 1e-3
                result[k] = math.log10(1 + abs(alpha_p / beta_p))
            return result

    def calculate_secure_capacity_of_user_k(self, k=2):
        """
        function used in update_channel_capacity to calculate the secure rate of user k
        """
        user = self.user_list[k]
        R_k_unsecure = user.capacity
        R_k_maxeavesdrop = max(self.eavesdrop_capacity_array[:, k])
        secrecy_rate= max(0, R_k_unsecure - R_k_maxeavesdrop)
        return secrecy_rate

    def get_system_action_dim(self):
        """
        function used in main function to get the dimention of actions
        """
        result = 0
        # 0 UAV movement
        result += 2
        # 1 RIS reflecting elements
        if self.if_with_RIS:
            result += self.RIS.ant_num   
        else:
            result += 0
        # 2 beamforming matrix dimention
        result += 2 * self.UAV.ant_num * self.user_num 
        return result

    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        综合信道 = BS + FAS, 维度 = bs_ant_num + fas_num_ports
        + FAS端口状态(1) + RIS调度状态(1)
        """
        result = 0
        total_ant = self.UAV.bs_ant_num + self.UAV.fas_num_ports
        result += 2 * (self.user_num + self.attacker_num) * total_ant
        if self.if_UAV_pos_state:
            result += 3
        result += 1  # FAS端口状态
        result += 1  # RIS调度状态
        return result
