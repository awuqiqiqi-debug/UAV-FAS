#%matplotlib inline
import numpy as np
import math
from entity import *
from channel import *
from math_tool import *
from datetime import datetime
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from render import Render
from data_manager import DataManager

np.random.seed(2)

######################################################
# 通信系统参数
fc = 28e9  # 载波频率 28GHz
C0 = 61  # 路径损耗基准常数 (dB)
P_max_dBm = 30  # 无人机最大发射功率上限 (dBm)
sigma_n = -114  # 接收端高斯噪声功率 (dBm)
sigma_s = 3  # 非完美CSI信道估计误差标准差 (dB)
L_sv = 3  # SV信道多径簇数量
delta_t = 0.1  # 单个优化时隙时长 (s)
Td = 1  # 信道相干时间 (s)
D_max = 0.25  # 单时隙无人机最大移动距离 (m)

######################################################
# 旋翼无人机能耗模型参数
# 基于经典旋翼无人机功率消耗模型: P = P0 + Pi + Ptip + Pbody
P_0 = 580.65  # 旋翼悬停桨叶基准剖面功率 (W) - 悬停固有损耗功率
P_i = 790.6715  # 悬停状态旋翼诱导功率 (W) - 下洗气流损耗功率
U_tip = 200  # 旋翼桨尖线速度 (m/s) - 用于计算桨叶附加风阻功耗
s = 0.05  # 旋翼桨叶实度 - 桨叶总面积/桨盘面积
d_0 = 0.3  # 无人机机身气动阻力系数
rho = 1.225  # 标准空气密度 (kg/m³)
A_r = 0.79  # 旋翼桨盘总面积 (m²)
delta_time = delta_t / 1000  # 时间步长转换

# 无人机物理参数
m = 1.3  # 无人机质量 (kg)
g = 9.81  # 重力加速度 (m/s²)
T_hover = m * g  # 悬停所需推力 (N)
v_0 = (T_hover / (A_r * 2 * rho)) ** 0.5  # 悬停诱导速度 (m/s)

def get_energy_consumption(v_t):
    """
    计算无人机能耗 (基于旋翼无人机功率消耗模型)
    P = P0 + Pi + Ptip + Pbody
    - P0: 桨叶剖面功率
    - Pi: 旋翼诱导功率
    - Ptip: 桨尖风阻功率
    - Pbody: 机身阻力功率
    """
    # 桨尖风阻功率 + 机身阻力功率
    energy_1 = P_0 + 3 * P_0 * (abs(v_t)) ** 2 / (U_tip ** 2) + 0.5 * d_0 * rho * s * A_r * (abs(v_t))**3
    # 旋翼诱导功率 (考虑前飞修正)
    energy_2 = P_i * ((1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5 - (abs(v_t) ** 2) / (2 * (v_0 **2))) ** 0.5
    # 总能耗 = 功率 × 时间
    energy = delta_time * (energy_1 + energy_2)
    return energy

ENERGY_MIN = get_energy_consumption(D_max)  # 最小能耗 (最大速度)
ENERGY_MAX = get_energy_consumption(0)  # 最大能耗 (悬停)

######################################################

class MiniSystem(object):
    """
    UAV-BS-FAS 一体化通信系统模型：
    - UAV集成基站(BS)和流体天线表面(FAS)功能（无人机搭载）
    - RIS反射单元阵列（地面固定）
    - 双路径信号传输：
      ① FAS发射信号 → RIS反射 → 用户（相长干涉增强）
      ② FAS发射信号 → RIS反射 → 攻击者（相消干涉干扰）
    - 支持多个用户和攻击者
    - 毫米波信道模型
    """
    def __init__(self, UAV_num = 1, user_num = 2, attacker_num = 1, fre = 28e9, \
                 bs_ant_num=4, fas_ant_num=12, ris_ant_num=64, if_dir_link = 1, if_with_FAS = True, \
                 if_move_users = False, if_movements = True, reverse_x_y = (False, False), \
                 if_UAV_pos_state = True, reward_design = 'ssr', project_name = None, step_num=100, existing_path=None):

        self.if_dir_link = if_dir_link
        self.if_with_FAS = if_with_FAS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.border = [(-50,50), (-20,60)]  # 水平空域边界
        self.ris_ant_num = ris_ant_num  # RIS反射单元数量
        
        # 初始化数据管理器
        self.data_manager = DataManager(file_path='./data', project_name = project_name, existing_path = existing_path, \
        store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity', 'secure_capacity', 'attaker_capacity','G_power', 'reward','UAV_movement'])
        
        # 初始化 UAV-BS-FAS 实体
        self.UAV_BS_FAS = UAV_BS_FAS(
            coordinate=self.data_manager.read_init_location('UAV', 0),
            bs_ant_num=bs_ant_num,
            fas_num_ports=fas_ant_num,
            max_movement_per_time_slot=D_max)  # 单时隙最大飞行距离0.25m
        
        self.UAV_BS_FAS.G = np.asmatrix(np.ones((self.UAV_BS_FAS.bs_ant_num, user_num), dtype=complex), dtype=complex)
        self.power_factor = 100
        self.UAV_BS_FAS.G_Pmax = np.trace(self.UAV_BS_FAS.G * self.UAV_BS_FAS.G.conj().T) * self.power_factor

        # 初始化用户
        self.user_list = []
        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = -114
            self.user_list.append(user)

        # 初始化攻击者
        self.attacker_list = []
        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros((user_num))
            attacker.noise_power = -114
            self.attacker_list.append(attacker)
        
        # 初始化RIS反射单元（地面固定）
        # RIS固定位置：(0, 50, 12.5)，64个反射单元（8x8阵列）
        ris_coordinate = np.array([0, 50, 12.5])  # RIS固定位置
        ris_coor_sys_z = np.array([0, 1, 0])  # 法线方向指向y轴(避免与[0,0,1]平行导致cross product为零)
        self.RIS = RIS(coordinate=ris_coordinate, coor_sys_z=ris_coor_sys_z, ant_num=ris_ant_num)
        
        self.eavesdrop_capacity_array= np.zeros((attacker_num, user_num))
        self.reward_design = reward_design
        self.step_num = step_num
        
        # 初始化信道
        # 直射链路：UAV-BS-FAS → 用户/攻击者
        self.h_U_k = []  # UAV到用户的直射信道
        self.h_U_p = []  # UAV到攻击者的直射信道

        for user_k in self.user_list:
            self.h_U_k.append(mmWave_channel(user_k, self.UAV_BS_FAS, fre))

        for attacker_p in self.attacker_list:
            self.h_U_p.append(mmWave_channel(attacker_p, self.UAV_BS_FAS, fre))

        # RIS反射链路信道
        # UAV → RIS 信道 (ris_ant_num, bs_ant_num)
        self.h_UR = mmWave_channel(self.RIS, self.UAV_BS_FAS, fre)
        # RIS → 用户k 信道 (1, ris_ant_num)
        self.h_R_k = []
        for user_k in self.user_list:
            self.h_R_k.append(mmWave_channel(self.RIS, user_k, fre))
        # RIS → 攻击者p 信道 (1, ris_ant_num)
        self.h_R_p = []
        for attacker_p in self.attacker_list:
            self.h_R_p.append(mmWave_channel(self.RIS, attacker_p, fre))

        # 初始化RIS双路径反射相位矩阵
        # Phi_signal：反射信号到用户的相位矩阵
        # Phi_jam：生成人工噪声干扰窃听者的相位矩阵
        self.RIS.Phi_signal = self.RIS.Phi   # 路径①：用户增强（信号反射）
        self.RIS.Phi_jam = self.RIS.Phi      # 路径②：攻击者干扰（人工噪声）

        # 初始化人工噪声功率（首次使用前必须存在）
        self.an_power = dB_to_normal(-114) * 1e-3

        self.update_channel_capacity()
        self.render_obj = Render(self)      

    def reset(self):
        """重置系统状态"""
        # 重置 UAV-BS-FAS
        self.UAV_BS_FAS.reset(coordinate=self.data_manager.read_init_location('UAV', 0))

        # 重置用户
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)

        # 重置攻击者
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)

        # 重置波束成形矩阵
        self.UAV_BS_FAS.G = np.asmatrix(np.ones((self.UAV_BS_FAS.bs_ant_num, self.user_num), dtype=complex), dtype=complex)
        self.UAV_BS_FAS.G_Pmax = np.trace(self.UAV_BS_FAS.G * self.UAV_BS_FAS.G.conj().T) * self.power_factor

        # 重置FAS发射矩阵 (关键: 非零初始化确保fas_gain>0)
        self.UAV_BS_FAS.F = np.asmatrix(
            np.ones((self.UAV_BS_FAS.fas_num_ports, 1), dtype=complex) / np.sqrt(self.UAV_BS_FAS.fas_num_ports)
        ) * math.pow(self.power_factor, 0.5)

        # 重置RIS反射相位矩阵
        self.RIS.Phi = np.asmatrix(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype = complex)
        
        # 重置时间索引
        self.render_obj.t_index = 0
        
        # 重置RIS反射相位矩阵
        self.RIS.Phi_signal = self.RIS.Phi  # 路径①：用户增强
        self.RIS.Phi_jam = self.RIS.Phi     # 路径②：攻击者干扰

        # 重置 CSI
        for h in self.h_U_k + self.h_U_p + [self.h_UR] + self.h_R_k + self.h_R_p:
            h.update_CSI()
        
        # 重置容量
        self.update_channel_capacity()

    def step(self, action_0 = 0, action_1 = 0, action_2 = 0, G = 0, Phi = 0, set_pos_x = 0, set_pos_y = 0, set_pos_z = 0):
        """执行一步模拟"""
        self.render_obj.t_index += 1

        # 更新用户位置
        if self.if_move_users and self.user_num > 1:
            self.user_list[0].update_coordinate(0.2, -1/2 * math.pi)
            self.user_list[1].update_coordinate(0.2, -1/2 * math.pi)

        # 更新 UAV-BS-FAS 位置
        if self.if_movements:
            move_x = action_0 * self.UAV_BS_FAS.max_movement_per_time_slot
            move_y = action_1 * self.UAV_BS_FAS.max_movement_per_time_slot
            move_z = action_2 * self.UAV_BS_FAS.max_movement_per_time_slot  # z轴移动
            v_t = (move_x ** 2 + move_y ** 2 + move_z ** 2) ** 0.5

            if self.reverse_x_y[0]:
                move_x = -move_x
            if self.reverse_x_y[1]:
                move_y = -move_y

            self.UAV_BS_FAS.coordinate[0] += move_x
            self.UAV_BS_FAS.coordinate[1] += move_y
            self.UAV_BS_FAS.coordinate[2] += move_z  # 更新z坐标
            self.data_manager.store_data([move_x, move_y, move_z], 'UAV_movement')
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV_BS_FAS.coordinate[0] = set_pos_x
            self.UAV_BS_FAS.coordinate[1] = set_pos_y

        # 更新信道 CSI（包括直射链路和RIS反射链路）
        for h in self.h_U_k + self.h_U_p + [self.h_UR] + self.h_R_k + self.h_R_p:
            h.update_CSI()
            
        if self.if_dir_link == 0:
            for h in self.h_U_k + self.h_U_p:
                h.channel_matrix = np.asmatrix(np.zeros(shape = np.shape(h.channel_matrix)), dtype=complex)

        # 更新波束成形矩阵、FAS发射矩阵和RIS反射相位
        # 动作解析：48维 = 16维BS波束 + 8维FAS波束 + 24维RIS相位
        # 16维BS波束: 4天线 × 2用户 × 2实虚 = 16
        # 8维FAS波束: 4端口 × 2实虚 = 8
        # 24维RIS相位: 12单元 × 2实虚 = 24

        # G = 前16维BS波束 + 8维FAS波束 = 24维
        bs_beam_action = G[:16]  # 16维: BS波束 (4×2 complex)
        fas_beam_action = G[16:24]  # 8维: FAS波束 (4×1 complex)

        # BS波束成形 (4天线 × 2用户)
        self.UAV_BS_FAS.G = convert_list_to_complex_matrix(bs_beam_action, (self.UAV_BS_FAS.bs_ant_num, self.user_num)) * math.pow(self.power_factor, 0.5)

        if self.if_with_FAS:
            # FAS流体天线：4个关键端口的波束成形 (4×1)
            self.UAV_BS_FAS.F = convert_list_to_complex_matrix(fas_beam_action,
                                                               (4, 1)) * math.pow(self.power_factor, 0.5)

            # 解析24维RIS相位：12维信号反射相位 + 12维人工噪声相位
            # 每12个控制相位重复填充至64个RIS单元
            ris_action = Phi[:24]
            ris_half = 12

            # 路径①：信号反射相位（12相位 → 64单元对角矩阵）
            signal_phases = ris_action[:ris_half]
            signal_expanded = np.tile(signal_phases, self.RIS.ant_num // ris_half + 1)[:self.RIS.ant_num]
            self.RIS.Phi = np.asmatrix(np.diag(np.exp(1j * signal_expanded * np.pi)), dtype=complex)
            self.RIS.Phi_signal = self.RIS.Phi

            # 路径②：人工噪声相位（12相位 → 64单元对角矩阵）
            jam_phases = ris_action[ris_half:]
            jam_expanded = np.tile(jam_phases, self.RIS.ant_num // ris_half + 1)[:self.RIS.ant_num]
            self.RIS.Phi_jam = np.asmatrix(np.diag(np.exp(1j * jam_expanded * np.pi)), dtype=complex)

        # 生成人工噪声向量（RIS本地生成，用于干扰窃听者）
        # AN功率与RIS放大增益相关
        self.an_power = dB_to_normal(-114) * 1e-3  # 基础噪声功率(mW)
        self.z_AN = (np.random.randn(self.RIS.ant_num, 1) + 1j * np.random.randn(self.RIS.ant_num, 1)) / np.sqrt(2)

        # 更新信道容量
        self.update_channel_capacity()
        
        # 存储系统状态
        self.store_current_system_sate()
        
        # 获取新状态
        new_state = self.observe()
        
        # 获取奖励
        reward = self.reward()

        # SEE 奖励设计: 能耗惩罚（固定权重）
        if self.reward_design == 'see':
            energy = get_energy_consumption(v_t)
            # 归一化能耗: 0=最大速度(最低能耗), 1=悬停(最高能耗)
            energy_norm = (energy - ENERGY_MIN) / (ENERGY_MAX - ENERGY_MIN + 1e-10)
            # 固定能耗惩罚: 每步最多-0.5，不再放大
            energy_penalty = -0.5 * energy_norm
            reward += energy_penalty

        # 边界惩罚
        done = False
        x, y = self.UAV_BS_FAS.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
            reward = -2.0
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -2.0
        
        self.data_manager.store_data([reward],'reward')
        return new_state, reward, done, []

    def reward(self):
        """计算奖励 - 归一化设计
        奖励范围: [-2, 12]
        """
        reward = 0

        # 功率惩罚: 固定-1
        P = np.trace(self.UAV_BS_FAS.G * self.UAV_BS_FAS.G.conj().T)
        if abs(P) > abs(self.UAV_BS_FAS.G_Pmax):
            return -1.0

        # ========== 位置奖励 [0, 10] ==========
        uav_pos = np.array(self.UAV_BS_FAS.coordinate[:2])
        ris_pos = np.array(self.RIS.coordinate[:2])
        dist_to_ris = np.linalg.norm(uav_pos - ris_pos)

        # 归一化位置奖励: 距离0m→10分, 距离50m→0分
        position_reward = max(0, 10.0 * (1.0 - dist_to_ris / 50.0))
        reward += position_reward

        # ========== 容量奖励 [0, 2] ==========
        total_secure = 0
        total_capacity = 0
        total_attacker_cap = 0

        for user in self.user_list:
            total_secure += max(user.secure_capacity, 0)
            total_capacity += user.capacity

        for attacker in self.attacker_list:
            total_attacker_cap += np.mean(attacker.capacity)

        # 安全容量奖励 [0, 1]
        reward += min(total_secure, 1.0)

        # 用户容量奖励 [0, 1]
        reward += min(total_capacity, 1.0)

        # 窃听者惩罚 [-1, 0]
        reward -= min(total_attacker_cap, 1.0)

        return reward
    
    def observe(self):
        """获取系统状态观测 (42维)
        状态包含：
        1. 用户有效信道实部+虚部 (2*K*N_BS = 16维)
        2. 窃听者有效信道实部+虚部 (2*P*N_BS = 8维)
        3. UAV位置坐标 (3维)
        4. 系统状态信息 (15维)
        """
        comprehensive_channel_elements_list = []
        for entity in self.user_list + self.attacker_list:
            tmp_list = list(np.array(np.reshape(entity.comprehensive_channel, (1,-1)))[0])
            comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list))

        UAV_position_list = []
        if self.if_UAV_pos_state:
            UAV_position_list = list(self.UAV_BS_FAS.coordinate)

        # 系统状态信息 (15维)
        system_state = [
            self.UAV_BS_FAS.bs_ant_num / 8.0,        # BS天线数归一化
            self.UAV_BS_FAS.fas_num_ports / 16.0,     # FAS端口数归一化
            self.RIS.ant_num / 64.0,                   # RIS单元数归一化
            self.user_num / 4.0,                       # 用户数归一化
            self.attacker_num / 2.0,                   # 窃听者数归一化
            2.2 / 4.0,                                 # αur归一化
            3.5 / 4.0,                                 # αul归一化
            2.8 / 4.0,                                 # αr归一化
            (sigma_n + 120) / 40.0,                    # 噪声功率归一化
            P_max_dBm / 40.0,                          # 最大功率归一化
            D_max / 1.0,                               # 最大移动距离归一化
            delta_t / 1.0,                             # 时隙时长归一化
            self.UAV_BS_FAS.coordinate[2] / 100.0,     # UAV高度归一化
            1.0,                                        # 标志位
            0.0,                                        # 保留
        ]

        return comprehensive_channel_elements_list + UAV_position_list + system_state

    def store_current_system_sate(self):
        """存储当前系统状态"""
        # 存储波束成形矩阵
        row_data = list(np.array(np.reshape(self.UAV_BS_FAS.G, (1, -1)))[0,:])
        self.data_manager.store_data(row_data, 'beamforming_matrix')
        
        # 存储RIS反射相位矩阵
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1,-1)))[0,:])      
        self.data_manager.store_data(row_data, 'reflecting_coefficient')
        
        # 存储 UAV 状态
        row_data = list(self.UAV_BS_FAS.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')
        
        # 存储功率信息
        row_data = [np.trace(self.UAV_BS_FAS.G*self.UAV_BS_FAS.G.conj().T), self.UAV_BS_FAS.G_Pmax]
        self.data_manager.store_data(row_data, 'G_power')
        
        # 存储用户容量
        row_data = []
        for user in self.user_list:
            row_data.append(user.capacity)
        self.data_manager.store_data(row_data, 'user_capacity')

        # 存储攻击者容量
        row_data = []
        for attacker in self.attacker_list:
            row_data.append(attacker.capacity)
        self.data_manager.store_data(row_data, 'attaker_capacity')

        # 存储安全容量
        row_data = []
        for user in self.user_list:
            row_data.append(user.secure_capacity)
        self.data_manager.store_data(row_data, 'secure_capacity')

    def update_channel_capacity(self):
        """更新用户和攻击者的信道容量"""
        # 更新攻击者容量
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
            attacker.comprehensive_channel = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
        
        # 更新用户容量
        for user in self.user_list:
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)

    def _ris_reflected_channel(self, h_R, ris_phi):
        """计算RIS反射路径: h_R (1, N_ris) @ Phi (N_ris, N_ris) @ h_UR.T (N_ris, N_bs) → (1, N_bs) row vector"""
        h_R_a = np.asarray(h_R)            # (1, N_ris)
        Phi_a = np.asarray(ris_phi)        # (N_ris, N_ris)
        h_UR_a = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_bs)
        return h_R_a @ Phi_a @ h_UR_a      # (1, N_bs)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """计算攻击者p的综合信道 (返回 column vector (N_bs, 1))"""
        h_d = np.asarray(self.h_U_p[p].channel_matrix)   # (1, N_bs)
        if self.if_with_FAS:
            fas_gain = np.mean(np.abs(self.UAV_BS_FAS.F)) if self.UAV_BS_FAS.F.size > 0 else 1
            h_R_p = np.asarray(self.h_R_p[p].channel_matrix)  # (1, N_ris)
            H_reflect = self._ris_reflected_channel(h_R_p, self.RIS.Phi_signal)  # (1, N_bs)
            H_eff = (h_d + H_reflect.T) * fas_gain  # (N_bs, 1)
            return H_eff
        else:
            return h_d

    def calculate_comprehensive_channel_of_user_k(self, k):
        """计算用户k的综合信道 (返回 column vector (N_bs, 1))"""
        h_d = np.asarray(self.h_U_k[k].channel_matrix)   # (N_bs, 1)
        if self.if_with_FAS:
            fas_gain = np.mean(np.abs(self.UAV_BS_FAS.F)) if self.UAV_BS_FAS.F.size > 0 else 1
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # (1, N_ris)
            H_reflect = self._ris_reflected_channel(h_R_k, self.RIS.Phi_signal)  # (1, N_bs)
            H_eff = (h_d + H_reflect.T) * fas_gain  # (N_bs, 1)
            return H_eff
        else:
            return h_d

    def calculate_capacity_of_user_k(self, k):
        """计算用户k的信道容量"""
        noise_power = self.user_list[k].noise_power

        if self.if_with_FAS:
            fas_gain = np.mean(np.abs(self.UAV_BS_FAS.F)) if self.UAV_BS_FAS.F.size > 0 else 1
            h_d = np.asarray(self.h_U_k[k].channel_matrix)  # (1, N_bs)
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # (1, N_ris)
            H_reflect = self._ris_reflected_channel(h_R_k, self.RIS.Phi_signal)  # (1, N_bs)
            H_eff = (h_d + H_reflect.T) * fas_gain  # (N_bs, 1)
        else:
            H_eff = np.asarray(self.h_U_k[k].channel_matrix)  # (N_bs, 1)

        G_k = np.asarray(self.UAV_BS_FAS.G[:, k])   # (N_bs, 1)

        if len(self.user_list) == 1:
            G_k_ = np.zeros((self.UAV_BS_FAS.bs_ant_num, 1), dtype=complex)
        else:
            G_k_ = np.hstack((np.asarray(self.UAV_BS_FAS.G[:, 0:k]),
                              np.asarray(self.UAV_BS_FAS.G[:, k+1:])))

        alpha_k = float(abs((H_eff.conj().T @ G_k).item())) ** 2
        beta_k = float(np.linalg.norm(H_eff * G_k_)) ** 2 + dB_to_normal(noise_power) * 1e-3
        return math.log10(1 + alpha_k / beta_k)

    def calculate_capacity_array_of_attacker_p(self, p):
        """计算攻击者p对所有用户的窃听容量"""
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power

        if self.if_with_FAS:
            fas_gain = np.mean(np.abs(self.UAV_BS_FAS.F)) if self.UAV_BS_FAS.F.size > 0 else 1
            h_d = np.asarray(self.h_U_p[p].channel_matrix)  # (1, N_bs)
            h_R_p = np.asarray(self.h_R_p[p].channel_matrix)  # (1, N_ris)
            H_reflect = self._ris_reflected_channel(h_R_p, self.RIS.Phi_signal)  # (1, N_bs)
            H_eff = (h_d + H_reflect.T) * fas_gain  # (N_bs, 1)
            jam_ch = h_R_p @ np.asarray(self.RIS.Phi_jam)  # (1, N_ris)
            an_power_eff = float(np.linalg.norm(jam_ch)) ** 2 * self.an_power
        else:
            H_eff = np.asarray(self.h_U_p[p].channel_matrix)  # (N_bs, 1)
            an_power_eff = 0

        if K == 1:
            G_k = np.asarray(self.UAV_BS_FAS.G)       # (N_bs, 1)
            G_k_ = np.zeros((self.UAV_BS_FAS.bs_ant_num, 1), dtype=complex)
        else:
            result = np.zeros(K)
            for k in range(K):
                G_k = np.asarray(self.UAV_BS_FAS.G[:, k])   # (N_bs, 1)
                G_k_ = np.hstack((np.asarray(self.UAV_BS_FAS.G[:, 0:k]),
                                  np.asarray(self.UAV_BS_FAS.G[:, k+1:])))
                alpha_p = float(abs((H_eff.conj().T @ G_k).item())) ** 2
                beta_p = float(np.linalg.norm(H_eff * G_k_)) ** 2 + dB_to_normal(noise_power) * 1e-3 + an_power_eff
                result[k] = math.log10(1 + alpha_p / beta_p)
            return result

        alpha_p = float(abs((H_eff.conj().T @ G_k).item())) ** 2
        beta_p = float(np.linalg.norm(H_eff * G_k_)) ** 2 + dB_to_normal(noise_power) * 1e-3 + an_power_eff
        return np.array([math.log10(1 + alpha_p / beta_p)])

    def calculate_secure_capacity_of_user_k(self, k=2):
        """计算用户k的安全容量"""
        user = self.user_list[k]
        R_k_unsecure = float(user.capacity)
        R_k_maxeavesdrop = float(np.max(self.eavesdrop_capacity_array[:, k]))
        diff = R_k_unsecure - R_k_maxeavesdrop
        secrecy_rate = diff if diff > 0.0 else 0.0
        return secrecy_rate

    def get_system_action_dim(self):
        """获取动作维度 (Agent 1)
        48维 = 16维BS波束 + 8维FAS波束 + 24维RIS相位
        RIS 24维: 12单元实部 + 12单元虚部 → 扩展填充至64单元
        """
        result = 0
        if self.if_with_FAS:
            result += 2 * self.UAV_BS_FAS.bs_ant_num * self.user_num  # BS波束: 16维
            result += 2 * 4  # FAS波束: 8维
            result += 2 * 12  # RIS相位: 24维 (12单元×2)
        return result

    def get_system_state_dim(self):
        """获取状态维度 (Agent 1)
        根据用户要求：42维 = 全链路CSI + 系统状态
        包含：用户信道(实虚) + 窃听者信道(实虚) + UAV位置(3) + 系统参数(15)
        """
        result = 0
        # 用户和窃听者的综合信道（实部+虚部）
        result += 2 * (self.user_num + self.attacker_num) * self.UAV_BS_FAS.bs_ant_num
        # UAV位置坐标
        if self.if_UAV_pos_state:
            result += 3
        # 系统状态：BS天线数、FAS端口数、RIS单元数、用户数、窃听者数等
        result += 15  # 15维系统状态信息
        return result

    def get_uav_local_state_dim(self):
        """获取无人机本地局部状态维度 (Agent 2)
        包含：UAV坐标(3) + 用户位置(2*3=6) = 9维
        """
        return 3 + self.user_num * 3  # 9维

    def observe_uav_local(self):
        """获取无人机本地局部状态观测 (Agent 2)
        包含：
        1. UAV当前位置坐标 (3维)
        2. 所有用户位置坐标 (K*3维)
        """
        state = []

        # UAV坐标 (3维)
        state.extend(list(self.UAV_BS_FAS.coordinate))

        # 用户位置 (K*3维)
        for user in self.user_list:
            state.extend(list(user.coordinate))

        return state