#%matplotlib inline
import numpy as np
import math
from src.envs.entity import UAV_BS_FAS, RIS, User, Attacker
from src.envs.channel import mmWave_channel
from src.envs.math_tool import dB_to_normal, normal_to_dB, map_to, convert_list_to_complex_matrix, diag
from datetime import datetime
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from src.utils.renderer import Render
from src.utils.data_manager import DataManager

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
                 fas_ant_num=12, ris_ant_num=64, if_dir_link = 1, if_with_FAS = True, \
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
        self.border = [(-50,50), (-50,50)]  # UAV飞行边界: X:-50~50, Y:-50~50
        self.ris_ant_num = ris_ant_num  # RIS反射单元数量
        
        # 初始化数据管理器
        self.data_manager = DataManager(file_path='./data', project_name = project_name, existing_path = existing_path, \
        store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity', 'secure_capacity', 'attaker_capacity','F_power', 'reward','UAV_movement', 'RIS_signal_phase', 'RIS_jam_phase'])
        
        # 初始化 UAV-BS-FAS 实体 (FAS作为唯一发射天线)
        self.UAV_BS_FAS = UAV_BS_FAS(
            coordinate=self.data_manager.read_init_location('UAV', 0),
            fas_num_ports=fas_ant_num,
            max_movement_per_time_slot=D_max)  # 单时隙最大飞行距离0.25m

        self.power_factor = 100
        self.UAV_BS_FAS.F = 1.0  # FAS增益（标量）
        self.UAV_BS_FAS.F_Pmax = P_max_dBm  # 最大功率 (dBm)

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
        # UAV → RIS 信道 (ris_ant_num, fas_num_ports)
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

        # 重置FAS增益（标量）
        self.UAV_BS_FAS.F = 1.0
        self.UAV_BS_FAS.F_Pmax = P_max_dBm

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

        # 保存当前动作供奖励函数使用
        self._last_action_0 = action_0
        self._last_action_1 = action_1

        # ========== 速度约束运动模型 ==========
        # v_max=1.0m/s, dt=0.1s, 单步最大位移=0.1m
        V_MAX = 1.0   # 最大水平速度 (m/s)
        DT = 0.1      # 仿真步长 (s)

        if self.if_movements:
            # Actor输出速度 [-1,1] → 实际速度 [-v_max, v_max]
            vx = np.clip(action_0, -1, 1) * V_MAX
            vy = np.clip(action_1, -1, 1) * V_MAX

            # 速度 → 位移
            move_x = vx * DT
            move_y = vy * DT

            # 边界软约束：接近边界时减速
            x, y = self.UAV_BS_FAS.coordinate[0], self.UAV_BS_FAS.coordinate[1]
            border_margin = 5  # 边界缓冲区
            if x > self.border[0][1] - border_margin:
                move_x = min(move_x, 0)
            elif x < self.border[0][0] + border_margin:
                move_x = max(move_x, 0)
            if y > self.border[1][1] - border_margin:
                move_y = min(move_y, 0)
            elif y < self.border[1][0] + border_margin:
                move_y = max(move_y, 0)

            if self.reverse_x_y[0]:
                move_x = -move_x
            if self.reverse_x_y[1]:
                move_y = -move_y

            self.UAV_BS_FAS.coordinate[0] += move_x
            self.UAV_BS_FAS.coordinate[1] += move_y
            self.UAV_BS_FAS.coordinate[2] = 50.0  # 固定高度50m
            v_t = math.sqrt(move_x ** 2 + move_y ** 2)
            self.data_manager.store_data([move_x, move_y, 0], 'UAV_movement')
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

        # 更新FAS波束成形矩阵和RIS反射相位
        # 动作解析：38维 = 13维FAS (12端口选择 + 1增益) + 25维RIS (1β + 24相位)
        # G: 前12维为端口选择softmax，第13维为F增益
        # Phi: 第1维为β，后24维为RIS相位

        if self.if_with_FAS:
            # 端口选择：RL直接从12个端口中选择（argmax）
            port_logits = G[:self.UAV_BS_FAS.fas_num_ports]  # 12维
            self.UAV_BS_FAS.fas_active_port = int(np.argmax(port_logits))

            # FAS增益（标量）
            self.UAV_BS_FAS.F = 1.0 + (G[self.UAV_BS_FAS.fas_num_ports] + 1.0) / 2.0 * 2.0  # [1, 3]

            # ====== RIS有源放大 + 相位优化（参考 ris_functions-jin） ======
            # 放大增益: β_i = sqrt(1 + 10*rand), 功率增益 β² ∈ [1, 11] 倍
            # 信号放大 → 放大给用户; 干扰放大 → 干扰窃听者
            # 两者共享同一组 β, 但通过不同相位矩阵分别作用

            # 从动作中提取放大增益标量 (第1维), 映射到 [1, sqrt(11)]
            BETA_MAX = 11.0  # 参考文件: β² = 1 + 10*rand → max=11
            ris_beta_raw = float(Phi[0]) if len(Phi) > 0 else 0.0
            ris_beta = 1.0 + (ris_beta_raw + 1.0) / 2.0 * (np.sqrt(BETA_MAX) - 1.0)  # [1, sqrt(11)] ≈ [1, 3.317]

            # 从动作中提取相位 (第2~25维, 共24维)
            ris_action = Phi[1:25] if len(Phi) >= 25 else Phi[1:]
            signal_phases = ris_action[:12]
            jam_phases = ris_action[12:24] if len(ris_action) >= 24 else ris_action[12:]

            # 自适应分配：窃听者信道强→多干扰，用户信道强→多反射
            user_gains = [np.sum(np.abs(np.asarray(self.h_R_k[u.index].channel_matrix))**2) for u in self.user_list]
            avg_user_gain = np.mean(user_gains) if user_gains else 1
            eve_gains = [np.sum(np.abs(np.asarray(self.h_R_p[a.index].channel_matrix))**2) for a in self.attacker_list]
            avg_eve_gain = np.mean(eve_gains) if eve_gains else 0
            ratio = avg_eve_gain / (avg_user_gain + 1e-10)
            jam_ratio = min(0.6, 0.3 + 0.3 * min(1.0, ratio))
            jam_elements = int(self.RIS.ant_num * jam_ratio)
            reflect_elements = self.RIS.ant_num - jam_elements

            # 信号反射相位 (反射元件 → 放大给用户)
            signal_expanded = np.tile(signal_phases, reflect_elements // 12 + 1)[:reflect_elements]
            signal_full = np.zeros(self.RIS.ant_num, dtype=complex)
            signal_full[:reflect_elements] = np.exp(1j * signal_expanded * np.pi)
            Phi_signal_mat = np.asmatrix(np.diag(signal_full), dtype=complex)

            # 人工噪声相位 (干扰元件 → 干扰窃听者)
            jam_expanded = np.tile(jam_phases, jam_elements // 12 + 1)[:jam_elements]
            jam_full = np.zeros(self.RIS.ant_num, dtype=complex)
            jam_full[reflect_elements:reflect_elements+jam_elements] = np.exp(1j * jam_expanded * np.pi)
            Phi_jam_mat = np.asmatrix(np.diag(jam_full), dtype=complex)

            # 有源放大: β × Φ (参考 ris_functions-jin: beta[i] = sqrt(1+10*rand))
            # 信号路径: β × Φ_signal → 放大用户信号
            # 干扰路径: β × Φ_jam → 放大干扰噪声
            beta_signal = np.asmatrix(np.diag(np.full(reflect_elements, ris_beta)), dtype=complex)
            beta_jam = np.asmatrix(np.diag(np.full(jam_elements, ris_beta)), dtype=complex)

            # 构建完整放大矩阵 (64×64对角)
            amp_full = np.zeros((self.RIS.ant_num, self.RIS.ant_num), dtype=complex)
            amp_full[:reflect_elements, :reflect_elements] = np.asarray(beta_signal)
            amp_full[reflect_elements:reflect_elements+jam_elements, reflect_elements:reflect_elements+jam_elements] = np.asarray(beta_jam)
            amp_diag = np.asmatrix(amp_full, dtype=complex)

            # 应用放大: θ = β × Φ
            self.RIS.Phi_signal = amp_diag @ Phi_signal_mat  # 信号: 放大给用户
            self.RIS.Phi_jam = amp_diag @ Phi_jam_mat        # 干扰: 放大干扰窃听者
            self.RIS.Phi = self.RIS.Phi_signal

            # RIS功率约束 (参考 FrisModule: limit_power)
            # 总功率 = 反射功率 + 干扰功率 + 热噪声 ≤ P_RIS_MAX
            P_RIS_MAX_dBm = 30  # RIS最大功率预算 30 dBm = 1W
            P_RIS_MAX_mW = 10 ** (P_RIS_MAX_dBm / 10)  # 1000 mW
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            # 使用活跃端口的信道
            active_port = self.UAV_BS_FAS.fas_active_port
            h_UR_active = h_UR[:, active_port:active_port+1]  # (N_ris, 1)

            # 反射功率: ||θ_R @ h_UR_active||² × F²
            P_reflect = 0
            Phi_R = np.asarray(self.RIS.Phi_signal)
            P_reflect = np.linalg.norm(Phi_R @ h_UR_active) ** 2 * self.UAV_BS_FAS.F ** 2

            # 干扰功率: ||θ_J @ h_UR_active||² × F²
            P_jam = 0
            Phi_J = np.asarray(self.RIS.Phi_jam)
            P_jam = np.linalg.norm(Phi_J @ h_UR_active) ** 2 * self.UAV_BS_FAS.F ** 2

            # 热噪声功率
            P_noise = (np.linalg.norm(Phi_R, 'fro') ** 2 + np.linalg.norm(Phi_J, 'fro') ** 2) * dB_to_normal(-114) * 1e-3

            P_RIS_total = P_reflect + P_jam + P_noise

            if P_RIS_total > P_RIS_MAX_mW:
                scale = np.sqrt(P_RIS_MAX_mW / (P_RIS_total + 1e-10))
                self.RIS.Phi_signal *= scale
                self.RIS.Phi_jam *= scale
                self.RIS.Phi = self.RIS.Phi_signal

            # 记录实际放大倍数
            self.RIS.amplification_gains = np.full(self.RIS.ant_num, ris_beta)

        # 人工噪声功率 — 参考FrisModule: P_J = 干扰功率参数
        # FrisModule: 干扰功率 = ||θ_J @ h_rp @ H_br @ G||²
        # 当前代码: an_power_eff = ||jam_ch||² × an_power
        # 所以 an_power 应该是基础干扰功率 (不含信道增益), 由 P_J 决定
        P_J_dBm = -50  # 干扰功率 -50 dBm = 0.00001 mW (中等干扰)
        self.an_power = 10 ** (P_J_dBm / 10)  # 1 mW
        self.z_AN = (np.random.randn(self.RIS.ant_num, 1) + 1j * np.random.randn(self.RIS.ant_num, 1)) / np.sqrt(2)

        # 更新信道容量
        self.update_channel_capacity()
        
        # 存储系统状态
        self.store_current_system_sate()
        
        # 获取新状态
        new_state = self.observe()
        
        # 获取奖励 (已在reward()中包含能耗惩罚和边界检查)
        reward = self.reward()

        # 边界检查 (轻度惩罚, 不覆盖奖励)
        done = False
        x, y = self.UAV_BS_FAS.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
            reward -= 0.3  # 轻度边界惩罚
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward -= 0.3  # 轻度边界惩罚
        
        self.data_manager.store_data([reward],'reward')

        # 存储RIS相位数据
        if self.if_with_FAS:
            # RIS信号反射相位 (归一化到[-1,1])
            signal_phase = np.angle(np.diag(np.asarray(self.RIS.Phi_signal))).tolist()
            self.data_manager.store_data([np.mean(signal_phase)/math.pi], 'RIS_signal_phase')
            # RIS干扰相位 (归一化到[-1,1])
            jam_phase = np.angle(np.diag(np.asarray(self.RIS.Phi_jam))).tolist()
            self.data_manager.store_data([np.mean(jam_phase)/math.pi], 'RIS_jam_phase')

        return new_state, reward, done, []

    def reward(self):
        """计算奖励 — 对应论文 Eq.(11)
        r = tanh(Σ R_k^sec - c1*p_m - c2*p_r - c4*p_e)

        设计依据:
        - Σ R_k^sec: 所有用户保密速率之和 (论文主目标)
        - p_m: 发射功率超限惩罚 (论文 c1*p_m)
        - p_r: 最低安全速率保障 (论文 c2*p_r)
        - p_e: 能耗惩罚, 仅在 ΣR^sec≥0 时生效 (论文 Eq.12: c4*p_e)
        - tanh归一化: 将奖励限制在[-1,1], 稳定critic的Q值估计
        """
        import math

        # === 1. 保密速率 SSR — 论文 Eq.(11): Σ R_k^sec ===
        total_secrecy = 0
        user_secrecies = []
        for user in self.user_list:
            secrecy_k = max(0, user.capacity - max(self.eavesdrop_capacity_array[:, user.index]))
            total_secrecy += secrecy_k
            user_secrecies.append(secrecy_k)

        # === 2. FAS端口保密增益 — 端口选择质量 ===
        # 衡量活跃端口对用户的信号强度 vs 对窃听者的信号强度
        R_fas = 0
        active_port = self.UAV_BS_FAS.fas_active_port
        if len(self.attacker_list) > 0:
            fas_risks = []
            for user in self.user_list:
                # 活跃端口到用户的信道增益
                g_user = abs(np.asarray(self.h_U_k[user.index].channel_matrix)[active_port, 0]) ** 2
                # 活跃端口到窃听者的信道增益
                g_eve = abs(np.asarray(self.h_U_p[self.attacker_list[0].index].channel_matrix)[active_port, 0]) ** 2
                if g_user > 0:
                    fas_risks.append(max(0, (g_user - g_eve) / (g_user + 1e-10)))
            if fas_risks:
                R_fas = np.mean(fas_risks)

        # === 3. 功率约束惩罚 — 论文 c1*p_m ===
        P = self.UAV_BS_FAS.F ** 2  # F是标量增益
        p_m = max(0, P - 10 ** (self.UAV_BS_FAS.F_Pmax / 10)) / (10 ** (self.UAV_BS_FAS.F_Pmax / 10) + 1e-10)

        # === 4. 最低安全速率惩罚 — 论文 c2*p_r ===
        R_th = 0.01
        p_r = 0
        for user in self.user_list:
            secrecy_k = max(0, user.capacity - max(self.eavesdrop_capacity_array[:, user.index]))
            if secrecy_k < R_th:
                p_r += (R_th - secrecy_k) / R_th

        # === 5. 能耗惩罚 — 论文 Eq.(12): c4*p_e ===
        # 论文定义: p_e = 0 (若 ΣR^sec < 0), 否则 p_e = 0.1 * ΣR^sec * E_p_norm
        v_t = getattr(self.UAV_BS_FAS, 'v_t', 0)
        E_p = get_energy_consumption(v_t)
        E_p_norm = (E_p - ENERGY_MIN) / (ENERGY_MAX - ENERGY_MIN + 1e-10)
        E_p_norm = max(0, min(1, E_p_norm))
        if total_secrecy < 0:
            p_e = 0.0
        else:
            p_e = 0.1 * total_secrecy * E_p_norm

        # === 组合奖励 — 论文 Eq.(11) ===
        raw_reward = (total_secrecy            # 论文: Σ R_k^sec
                      + 0.3 * R_fas            # FAS端口保密增益（辅助信号）
                      - p_m                    # 论文: c1*p_m
                      - 2.0 * p_r              # 论文: c2*p_r
                      - p_e)                   # 论文: c4*p_e (Eq.12)

        # tanh 归一化到 [-1, 1] — 论文 Eq.(11)
        return np.tanh(raw_reward)

    def find_optimal_uav_position(self, grid_step=10):
        """
        网格搜索最优UAV位置，最大化总安全容量
        返回: (最优x, 最优y, 最大安全容量)
        """
        best_pos = None
        best_sec = -float('inf')

        # 保存原始位置
        original_pos = self.UAV_BS_FAS.coordinate.copy()

        for x in range(self.border[0][0], self.border[0][1] + 1, grid_step):
            for y in range(self.border[1][0], self.border[1][1] + 1, grid_step):
                # 设置临时位置
                self.UAV_BS_FAS.coordinate[0] = x
                self.UAV_BS_FAS.coordinate[1] = y

                # 更新信道
                for h in self.h_U_k + self.h_U_p + [self.h_UR] + self.h_R_k + self.h_R_p:
                    h.update_CSI()
                if self.if_dir_link == 0:
                    for h in self.h_U_k + self.h_U_p:
                        h.channel_matrix = np.asmatrix(np.zeros(shape=h.channel_matrix.shape), dtype=complex)

                # 计算安全容量
                self.update_channel_capacity()
                total_sec = sum(max(0, u.capacity - max(self.eavesdrop_capacity_array[:, u.index]))
                               for u in self.user_list)

                if total_sec > best_sec:
                    best_sec = total_sec
                    best_pos = (x, y)

        # 恢复原始位置
        self.UAV_BS_FAS.coordinate = original_pos

        return best_pos, best_sec
    
    def observe(self):
        """获取系统状态观测 (89维)
        状态包含：
        1. 用户有效信道实部+虚部 (2*K*N_FAS)
        2. 窃听者有效信道实部+虚部 (2*P*N_FAS)
        3. UAV位置坐标 (3维)
        4. 系统状态信息 (14维)
        """
        # 各端口到用户和窃听者的信道（用于端口选择）
        port_channel_list = []
        for entity in self.user_list + self.attacker_list:
            # 信道形状: (N_FAS, 1) = (12, 1)，取所有端口的信道增益
            channel = np.asarray(entity.comprehensive_channel_raw).flatten()  # (12,) complex
            port_channel_list += list(np.real(channel)) + list(np.imag(channel))

        UAV_position_list = []
        if self.if_UAV_pos_state:
            UAV_position_list = list(self.UAV_BS_FAS.coordinate)

        # 系统状态信息 (14维)
        system_state = [
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
            self.UAV_BS_FAS.fas_active_port / 12.0,    # 当前端口归一化
            0.0,                                        # 保留
        ]

        return port_channel_list + UAV_position_list + system_state

    def store_current_system_sate(self):
        """存储当前系统状态"""
        # 存储FAS信息（活跃端口和增益）
        row_data = [self.UAV_BS_FAS.fas_active_port, self.UAV_BS_FAS.F]
        self.data_manager.store_data(row_data, 'beamforming_matrix')

        # 存储RIS反射相位矩阵
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1,-1)))[0,:])
        self.data_manager.store_data(row_data, 'reflecting_coefficient')

        # 存储 UAV 状态
        row_data = list(self.UAV_BS_FAS.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')

        # 存储功率信息
        row_data = [self.UAV_BS_FAS.F ** 2, self.UAV_BS_FAS.F_Pmax]
        self.data_manager.store_data(row_data, 'F_power')
        
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
        # 存储各端口的原始信道（用于observe）
        for entity in self.user_list + self.attacker_list:
            if entity.type == 'user':
                entity.comprehensive_channel_raw = np.asarray(self.h_U_k[entity.index].channel_matrix)  # (N_FAS, 1)
            else:
                entity.comprehensive_channel_raw = np.asarray(self.h_U_p[entity.index].channel_matrix)  # (N_FAS, 1)

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
        """
        计算有源RIS反射路径 (参考ris_functions-jin-xinhaofngda.py)
        h_R (1, N_ris) @ diag(beta_i) @ Phi (N_ris, N_ris) @ h_UR.T (N_ris, N_FAS) → (1, N_FAS)

        有源RIS: 每个反射单元有放大增益 beta_i = sqrt(1 + 10*rand)
        """
        h_R_a = np.asarray(h_R)            # (1, N_ris)
        Phi_a = np.asarray(ris_phi)        # (N_ris, N_ris)
        h_UR_a = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)

        # 有源RIS放大增益 (对角矩阵)
        amp_gains = np.diag(self.RIS.amplification_gains)  # (N_ris, N_ris)

        # 有源反射: h_R @ diag(beta) @ Phi @ h_UR
        return h_R_a @ amp_gains @ Phi_a @ h_UR_a      # (1, N_FAS)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """计算攻击者p的综合信道 (标量，基于活跃端口)"""
        active_port = self.UAV_BS_FAS.fas_active_port
        if self.if_with_FAS:
            # 直传路径：活跃端口到攻击者 (channel shape: (N_FAS, 1), 取第active_port行)
            h_d = np.asarray(self.h_U_p[p].channel_matrix)[active_port, 0]  # 标量
            # RIS反射路径：RIS → 活跃端口
            h_R_p = np.asarray(self.h_R_p[p].channel_matrix)  # (1, N_ris)
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_UR_active = h_UR[:, active_port]  # (N_ris,)
            Phi_signal = np.asarray(self.RIS.Phi_signal)
            amp_gains = np.diag(self.RIS.amplification_gains)
            H_reflect = h_R_p @ amp_gains @ Phi_signal @ h_UR_active  # 标量
            H_eff = h_d + H_reflect
            return np.array([[H_eff]])  # (1, 1)
        else:
            return np.asarray(self.h_U_p[p].channel_matrix)[active_port:active_port+1, :]

    def calculate_comprehensive_channel_of_user_k(self, k):
        """计算用户k的综合信道 (标量，基于活跃端口)"""
        active_port = self.UAV_BS_FAS.fas_active_port
        if self.if_with_FAS:
            # 直传路径：活跃端口到用户 (channel shape: (N_FAS, 1), 取第active_port行)
            h_d = np.asarray(self.h_U_k[k].channel_matrix)[active_port, 0]  # 标量
            # RIS反射路径：RIS → 活跃端口
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # (1, N_ris)
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_UR_active = h_UR[:, active_port]  # (N_ris,)
            Phi_signal = np.asarray(self.RIS.Phi_signal)
            amp_gains = np.diag(self.RIS.amplification_gains)
            H_reflect = h_R_k @ amp_gains @ Phi_signal @ h_UR_active  # 标量
            H_eff = h_d + H_reflect
            return np.array([[H_eff]])  # (1, 1)
        else:
            return np.asarray(self.h_U_k[k].channel_matrix)[active_port:active_port+1, :]

    def calculate_capacity_of_user_k(self, k):
        """计算用户k的信道容量"""
        noise_power = self.user_list[k].noise_power
        active_port = self.UAV_BS_FAS.fas_active_port

        if self.if_with_FAS:
            # 活跃端口的综合信道 (channel shape: (N_FAS, 1), 取第active_port行)
            h_d = np.asarray(self.h_U_k[k].channel_matrix)[active_port, 0]  # 标量
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # (1, N_ris)
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_UR_active = h_UR[:, active_port]  # (N_ris,)
            Phi_signal = np.asarray(self.RIS.Phi_signal)
            amp_gains = np.diag(self.RIS.amplification_gains)
            H_reflect = h_R_k @ amp_gains @ Phi_signal @ h_UR_active  # 标量
            H_eff = (h_d + H_reflect) * self.UAV_BS_FAS.F  # 标量 × F增益
        else:
            H_eff = np.asarray(self.h_U_k[k].channel_matrix)[active_port, 0] * self.UAV_BS_FAS.F

        # 等功率分配：所有用户共享活跃端口的功率
        alpha_k = abs(H_eff) ** 2
        # 干扰：其他用户的信号在同一端口传输
        beta_k = alpha_k * (len(self.user_list) - 1) / len(self.user_list) + dB_to_normal(noise_power) * 1e-3
        return math.log10(1 + alpha_k / beta_k)

    def calculate_capacity_array_of_attacker_p(self, p):
        """计算攻击者p对所有用户的窃听容量"""
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        active_port = self.UAV_BS_FAS.fas_active_port

        if self.if_with_FAS:
            # 活跃端口到攻击者的综合信道 (channel shape: (N_FAS, 1), 取第active_port行)
            h_d = np.asarray(self.h_U_p[p].channel_matrix)[active_port, 0]  # 标量
            h_R_p = np.asarray(self.h_R_p[p].channel_matrix)  # (1, N_ris)
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_UR_active = h_UR[:, active_port]  # (N_ris,)
            Phi_signal = np.asarray(self.RIS.Phi_signal)
            Phi_jam = np.asarray(self.RIS.Phi_jam)
            amp_gains = np.diag(self.RIS.amplification_gains)
            H_reflect = h_R_p @ amp_gains @ Phi_signal @ h_UR_active  # 标量
            H_eff = (h_d + H_reflect) * self.UAV_BS_FAS.F
            # 干扰噪声
            jam_ch = h_R_p @ Phi_jam @ h_UR_active  # 标量
            an_power_eff = abs(jam_ch) ** 2 * self.an_power
        else:
            H_eff = np.asarray(self.h_U_p[p].channel_matrix)[active_port, 0] * self.UAV_BS_FAS.F
            an_power_eff = 0

        # 攻击者窃听所有用户的信号
        alpha_p = abs(H_eff) ** 2
        beta_p = alpha_p * (K - 1) / K + dB_to_normal(noise_power) * 1e-3 + an_power_eff
        return np.array([math.log10(1 + alpha_p / beta_p)] * K)

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
        38维 = 12维端口选择 + 1维F增益 + 1维RIS放大增益 + 24维RIS相位
        """
        result = 0
        if self.if_with_FAS:
            result += self.UAV_BS_FAS.fas_num_ports  # 端口选择: 12维 (softmax)
            result += 1      # FAS增益 F: 1维
            result += 1      # RIS放大增益 β: 1维
            result += 2 * 12  # RIS相位: 24维 (12单元×2)
        return result

    def get_system_state_dim(self):
        """获取状态维度 (Agent 1)
        89维 = 各端口信道 + UAV位置(3) + 系统状态(14)
        包含：12端口×3实体×2(实虚) + UAV位置 + 系统参数
        """
        result = 0
        # 各端口到用户和窃听者的信道（实部+虚部）
        result += 2 * (self.user_num + self.attacker_num) * self.UAV_BS_FAS.fas_num_ports
        # UAV位置坐标
        if self.if_UAV_pos_state:
            result += 3
        # 系统状态：FAS端口数、RIS单元数、用户数、窃听者数等
        result += 14  # 14维系统状态信息
        return result

    def get_uav_local_state_dim(self):
        """获取无人机本地局部状态维度 (Agent 2)
        包含：UAV坐标(3) + 用户位置(K*3) + RIS位置(3) + 窃听者位置(3)
              + 用户信道容量(K) + 窃听者信道容量(1) = 18维
        """
        return 3 + self.user_num * 3 + 3 + 3 + self.user_num + 1  # 18维

    def observe_uav_local(self):
        """获取无人机本地局部状态观测 (Agent 2)
        包含：
        1. UAV当前位置坐标 (3维)
        2. 所有用户位置坐标 (K*3维)
        3. RIS位置 (3维)
        4. 窃听者位置 (3维)
        5. 用户信道容量 (K维) ← 新增
        6. 窃听者信道容量 (1维) ← 新增
        总计: 18维
        """
        state = []

        # UAV坐标 (3维)
        state.extend(list(self.UAV_BS_FAS.coordinate))

        # 用户位置 (K*3维)
        for user in self.user_list:
            state.extend(list(user.coordinate))

        # RIS位置 (3维)
        state.extend(list(self.RIS.coordinate))

        # 窃听者位置 (3维)
        for attacker in self.attacker_list:
            state.extend(list(attacker.coordinate))

        # 用户信道容量 (K维) - Agent 2需要知道当前位置的信道质量
        for user in self.user_list:
            state.append(float(user.capacity))

        # 窃听者信道容量 (1维) - Agent 2需要知道窃听者的信道质量
        if len(self.attacker_list) > 0:
            state.append(float(np.mean(self.attacker_list[0].capacity)))
        else:
            state.append(0.0)

        return state