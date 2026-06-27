# %matplotlib inline
import numpy as np
import math
import torch
import torch.nn.functional as F
from src.envs.entity import UAV_FAS, RIS, User, Attacker
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
delta_time = delta_t  # 时间步长 (s)，与delta_t=0.1s一致

# 无人机物理参数
m = 1.3  # 无人机质量 (kg)
g = 9.81  # 重力加速度 (m/s²)
T_hover = m * g  # 悬停所需推力 (N)
v_0 = (T_hover / (A_r * 2 * rho)) ** 0.5  # 悬停诱导速度 (m/s)


def get_energy_consumption(v_t, dt=None):
    """
    计算无人机能耗 (基于旋翼无人机功率消耗模型)
    P = P0 + Pi + Ptip + Pbody
    - v_t: 速度 (m/s)
    - dt: 时间步长 (s)，默认使用全局delta_t
    返回: 能耗 (J)
    """
    if dt is None:
        dt = delta_t
    # 桨尖风阻功率 + 机身阻力功率
    energy_1 = P_0 + 3 * P_0 * (abs(v_t)) ** 2 / (U_tip ** 2) + 0.5 * d_0 * rho * s * A_r * (abs(v_t)) ** 3
    # 旋翼诱导功率 (考虑前飞修正)
    energy_2 = P_i * ((1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5 - (abs(v_t) ** 2) / (2 * (v_0 ** 2))) ** 0.5
    # 总能耗 = 功率 × 时间
    energy = dt * (energy_1 + energy_2)
    return energy


ENERGY_MIN = get_energy_consumption(D_max)  # 最小能耗 (最大速度)
ENERGY_MAX = get_energy_consumption(0)  # 最大能耗 (悬停)


######################################################

class MiniSystem(object):
    """
    UAV-FAS 一体化通信系统模型：
    - UAV集成基站(BS)和流体天线表面(FAS)功能（无人机搭载）
    - RIS反射单元阵列（地面固定）
    - 双路径信号传输：
      ① FAS发射信号 → RIS反射 → 用户（相长干涉增强）
      ② FAS发射信号 → RIS反射 → 攻击者（相消干涉干扰）
    - 支持多个用户和攻击者
    - 毫米波信道模型
    """

    def __init__(self, UAV_num=1, user_num=2, attacker_num=1, fre=28e9, \
                 fas_ant_num=12, ris_ant_num=64, if_dir_link=1, if_with_FAS=True, \
                 if_move_users=False, if_movements=True, reverse_x_y=(False, False), \
                 if_UAV_pos_state=True, reward_design='ssr', project_name=None, step_num=100, existing_path=None, \
                 num_active_ports=2, total_episodes=1000):

        self.if_dir_link = if_dir_link
        self.if_with_FAS = if_with_FAS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.border = [(-50, 50), (-50, 50)]  # UAV飞行边界: X:-50~50, Y:-50~50
        self.ris_ant_num = ris_ant_num  # RIS反射单元数量
        self.num_active_ports = num_active_ports  # 同时激活的FAS端口数 (2~3)
        self.training = True  # 训练/推理模式切换
        # RIS干扰相位: Agent 100%自主控制，无启发式对准
        self.jam_align_weight = 0.0    # 0%启发式，100% Agent自主
        self.jam_align_decay = 1.0     # 不增长
        self.jam_align_max = 0.0       # 不引入启发式对准
        self.total_train_steps = total_episodes * step_num  # 总训练步数

        # 初始化数据管理器
        self.data_manager = DataManager(file_path='./data', project_name=project_name, existing_path=existing_path, \
                                        store_list=['beamforming_matrix', 'reflecting_coefficient', 'UAV_state',
                                                    'user_capacity', 'secure_capacity', 'attaker_capacity', 'F_power',
                                                    'reward', 'UAV_movement', 'RIS_signal_phase', 'RIS_jam_phase',
                                                    'FAS_active_port'])

        # 初始化 UAV-FAS 实体 (FAS作为唯一发射天线)
        self.UAV_FAS = UAV_FAS(
            coordinate=self.data_manager.read_init_location('UAV', 0),
            fas_num_ports=fas_ant_num,
            max_movement_per_time_slot=D_max)  # 单时隙最大飞行距离0.25m

        self.power_factor = 100
        self.UAV_FAS.F = 1.0  # FAS增益（标量）
        self.UAV_FAS.F_Pmax = P_max_dBm  # 最大功率 (dBm)

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
        # 从Excel文件读取RIS位置
        ris_coordinate = self.data_manager.read_init_location('RIS', 0)
        ris_norm_vec = self.data_manager.read_init_location('RIS_norm_vec', 0)
        ris_coor_sys_z = ris_norm_vec / np.linalg.norm(ris_norm_vec)  # 归一化法线方向
        self.RIS = RIS(coordinate=ris_coordinate, coor_sys_z=ris_coor_sys_z, ant_num=ris_ant_num,
                       Pr=30, P_J=30, beta=30, sigma=-100)  # Pr=30dBm, P_J=30dBm

        self.eavesdrop_capacity_array = np.zeros((attacker_num, user_num))
        self.reward_design = reward_design
        self.step_num = step_num

        # 初始化信道
        # 直射链路：UAV-FAS → 用户/攻击者
        self.h_U_k = []  # UAV到用户的直射信道
        self.h_U_p = []  # UAV到攻击者的直射信道

        for user_k in self.user_list:
            self.h_U_k.append(mmWave_channel(user_k, self.UAV_FAS, fre))

        for attacker_p in self.attacker_list:
            self.h_U_p.append(mmWave_channel(attacker_p, self.UAV_FAS, fre))

        # RIS反射链路信道
        # UAV → RIS 信道 (ris_ant_num, fas_num_ports)
        self.h_UR = mmWave_channel(self.RIS, self.UAV_FAS, fre)
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
        self.RIS.Phi_signal = self.RIS.Phi  # 路径①：用户增强（信号反射）
        self.RIS.Phi_jam = self.RIS.Phi  # 路径②：攻击者干扰（人工噪声）

        # 初始化人工噪声功率（首次使用前必须存在）
        self.an_power = dB_to_normal(-114) * 1e-3

        # 初始化用户级波束成形权重 (K, num_active_ports)
        # 每个用户在各活跃端口上的权重，softmax归一化
        self.user_beamforming_weights = np.ones((self.user_num, self.num_active_ports)) / self.num_active_ports

        self.update_channel_capacity()
        self.render_obj = Render(self)

    def reset(self):
        """重置系统状态"""
        # 重置 UAV-FAS
        self.UAV_FAS.reset(coordinate=self.data_manager.read_init_location('UAV', 0))

        # 重置用户
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)

        # 重置攻击者
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)

        # 重置FAS增益（标量）
        self.UAV_FAS.F = 1.0
        self.UAV_FAS.F_Pmax = P_max_dBm

        # 重置用户级波束成形权重 (K, num_active_ports)
        self.user_beamforming_weights = np.ones((self.user_num, self.num_active_ports)) / self.num_active_ports

        # 重置RIS反射相位矩阵
        self.RIS.Phi = np.asmatrix(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype=complex)

        # 重置时间索引
        self.render_obj.t_index = 0

        # 重置RIS反射相位矩阵
        self.RIS.Phi_signal = self.RIS.Phi  # 路径①：用户增强
        self.RIS.Phi_jam = self.RIS.Phi  # 路径②：攻击者干扰

        # 重置 CSI
        for h in self.h_U_k + self.h_U_p + [self.h_UR] + self.h_R_k + self.h_R_p:
            h.update_CSI()

        # 重置容量
        self.update_channel_capacity()

    def step(self, action_0=0, action_1=0, action_2=0, G=0, Phi=0, user_weights=None, set_pos_x=0, set_pos_y=0,
             set_pos_z=0):
        """执行一步模拟"""
        self.render_obj.t_index += 1

        # 更新用户位置
        if self.if_move_users and self.user_num > 1:
            self.user_list[0].update_coordinate(0.2, -1 / 2 * math.pi)
            self.user_list[1].update_coordinate(0.2, -1 / 2 * math.pi)

        # 保存当前动作供奖励函数使用
        self._last_action_0 = action_0
        self._last_action_1 = action_1

        # ========== 速度约束运动模型 ==========
        # v_max=5.0m/s, dt=0.1s, 单步最大位移=0.5m, 150步最大75m
        V_MAX = 5.0  # 最大水平速度 (m/s)，小型旋翼典型巡航速度
        DT = 0.1  # 仿真步长 (s)

        if self.if_movements:
            # Actor输出速度 [-1,1] → 实际速度 [-v_max, v_max]
            vx = np.clip(action_0, -1, 1) * V_MAX
            vy = np.clip(action_1, -1, 1) * V_MAX

            # 速度 → 位移
            move_x = vx * DT
            move_y = vy * DT

            # 边界软约束：接近边界时减速
            x, y = self.UAV_FAS.coordinate[0], self.UAV_FAS.coordinate[1]
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

            self.UAV_FAS.coordinate[0] += move_x
            self.UAV_FAS.coordinate[1] += move_y
            self.UAV_FAS.coordinate[2] = 50.0  # 固定高度50m
            v_t = math.sqrt(move_x ** 2 + move_y ** 2) / DT  # 实际速度 (m/s)
            self.UAV_FAS.v_t = v_t  # 保存速度到实体，供奖励函数使用
            self.data_manager.store_data([move_x, move_y, 0], 'UAV_movement')
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV_FAS.coordinate[0] = set_pos_x
            self.UAV_FAS.coordinate[1] = set_pos_y

        # 更新信道 CSI（包括直射链路和RIS反射链路）
        for h in self.h_U_k + self.h_U_p + [self.h_UR] + self.h_R_k + self.h_R_p:
            h.update_CSI()

        if self.if_dir_link == 0:
            for h in self.h_U_k + self.h_U_p:
                h.channel_matrix = np.asmatrix(np.zeros(shape=np.shape(h.channel_matrix)), dtype=complex)

        # 更新FAS波束成形矩阵和RIS反射相位
        # 动作解析：38维 = 13维FAS (12端口选择 + 1增益) + 25维RIS (1β + 24相位)
        # G: 前12维为端口选择softmax，第13维为F增益
        # Phi: 第1维为β，后24维为RIS相位

        if self.if_with_FAS:
            # ====== 端口选择：Gumbel-Softmax + Top-K (2~3端口) ======
            port_logits_tensor = torch.tensor(G[:self.UAV_FAS.fas_num_ports], dtype=torch.float)
            if self.training:
                # 课程学习: τ从1.0衰减到0.1，逐步从随机→确定性选择
                # 余弦退火: τ 从 0.8 缓慢降到 0.1
                if not hasattr(self, '_gumbel_tau'):
                    self._gumbel_tau = 0.8
                    self._gumbel_step = 0
                self._gumbel_step += 1
                progress = min(1.0, self._gumbel_step / max(1, self.total_train_steps))
                # 前50%训练保持高探索(τ=0.6~0.8)，后50%才衰减到0.1
                if progress < 0.5:
                    self._gumbel_tau = 0.6 + 0.2 * (1.0 - progress / 0.5)
                else:
                    decay_progress = (progress - 0.5) / 0.5
                    self._gumbel_tau = 0.6 - 0.5 * decay_progress
                port_probs = F.gumbel_softmax(port_logits_tensor, tau=self._gumbel_tau, hard=False)
                # 选择概率最高的K个端口
                _, topk_indices = torch.topk(port_probs, self.num_active_ports)
            else:
                # 推理时: 硬选择Top-K
                _, topk_indices = torch.topk(port_logits_tensor, self.num_active_ports)

            self.UAV_FAS.fas_active_ports = topk_indices.numpy().tolist()
            self.UAV_FAS.fas_active_port = self.UAV_FAS.fas_active_ports[0]  # 主端口(兼容)

            # FAS增益（标量），K个端口等功率分配
            F_total = 0.3 + (G[self.UAV_FAS.fas_num_ports] + 1.0) / 2.0 * 0.7  # [0.3, 1.0]
            # 功率硬约束: 总功率 F_total² ≤ P_max_mW
            P_max_linear = 10 ** (P_max_dBm / 10)  # 1000 mW
            if F_total ** 2 > P_max_linear:
                F_total = np.sqrt(P_max_linear)
            self.UAV_FAS.F = F_total / math.sqrt(self.num_active_ports)  # 每端口功率 = F_total² / K

            # ====== 用户级波束成形权重 (K×num_active_ports) ======
            if user_weights is not None and len(user_weights) == self.user_num * self.num_active_ports:
                user_weights_raw = np.array(user_weights).reshape(self.user_num, self.num_active_ports)
                # softmax归一化: 每个用户的权重和为1
                for k in range(self.user_num):
                    exp_w = np.exp(user_weights_raw[k] - np.max(user_weights_raw[k]))  # 数值稳定
                    self.user_beamforming_weights[k] = exp_w / (exp_w.sum() + 1e-10)
            # else: 保持默认均匀权重

            # ====== RIS有源放大 + 相位优化（Agent控制） ======
            # 动作解析: Phi[0]=β, Phi[1]=η(jam_ratio), Phi[2:14]=信号相位, Phi[14:26]=干扰相位

            # 从动作中提取放大增益标量 (第0维), 映射到 [1, sqrt(BETA_MAX)]
            BETA_MAX = 20.0  # β²最大20，ris_beta最大≈4.47，让RIS干扰足以压制窃听者
            ris_beta_raw = float(np.clip(Phi[0], -1.0, 1.0)) if len(Phi) > 0 else 0.0
            ris_beta = 1.0 + (ris_beta_raw + 1.0) / 2.0 * (np.sqrt(BETA_MAX) - 1.0)  # [1, 10]

            # 从动作中提取干扰比例 η (第1维), 映射到 [0.1, 0.5]
            # 扩大干扰范围，让Agent有足够自由度抑制窃听者
            jam_ratio_raw = float(np.clip(Phi[1], -1.0, 1.0)) if len(Phi) > 1 else 0.0
            jam_ratio = 0.1 + (jam_ratio_raw + 1.0) / 2.0 * 0.4  # [0.1, 0.5]
            jam_elements = int(self.RIS.ant_num * jam_ratio)
            jam_elements = max(1, min(jam_elements, self.RIS.ant_num - 1))  # 确保至少1个干扰元件，至少1个反射元件
            reflect_elements = self.RIS.ant_num - jam_elements

            # 从动作中提取相位 (第2~25维, 共24维)
            ris_action = Phi[2:26] if len(Phi) >= 26 else Phi[2:]
            signal_phases = ris_action[:12]
            jam_phases = ris_action[12:24] if len(ris_action) >= 24 else ris_action[12:]

            # 信号反射相位 (反射元件 → 放大给用户)
            signal_expanded = np.tile(signal_phases, reflect_elements // 12 + 1)[:reflect_elements]
            signal_full = np.zeros(self.RIS.ant_num, dtype=complex)
            signal_full[:reflect_elements] = np.exp(1j * signal_expanded * np.pi)
            Phi_signal_mat = np.asmatrix(np.diag(signal_full), dtype=complex)

            # 人工噪声相位 (干扰元件 → 干扰窃听者)
            # 课程学习: Agent相位 + 窃听者信道对齐相位的加权混合
            # 初始阶段: 对准窃听者80%，Agent学习20%
            # 后期阶段: Agent完全自主控制
            h_R_p_arr = np.asarray(self.h_R_p[0].channel_matrix).flatten()  # (N_ris,)
            h_UR_arr = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            active_port_local = self.UAV_FAS.fas_active_port
            h_UR_active = h_UR_arr[:, active_port_local]  # (N_ris,)
            eve_channel_phase = np.angle(h_R_p_arr * h_UR_active)  # 窃听者信道相位

            # 课程学习: 对准权重从0.2逐步增长到0.5
            growth = 2.0 - self.jam_align_decay  # 1.0001
            if self.jam_align_weight < self.jam_align_max:
                self.jam_align_weight = min(self.jam_align_max,
                                            self.jam_align_weight * growth)

            # Agent相位 + 对准相位的加权混合
            jam_agent_phase = np.tile(jam_phases, jam_elements // 12 + 1)[:jam_elements]  # Agent输出
            jam_align_phase = -eve_channel_phase[reflect_elements:reflect_elements + jam_elements]  # 对准窃听者
            jam_mixed_phase = (1 - self.jam_align_weight) * jam_agent_phase + self.jam_align_weight * jam_align_phase

            jam_full = np.zeros(self.RIS.ant_num, dtype=complex)
            jam_full[reflect_elements:reflect_elements + jam_elements] = np.exp(1j * jam_mixed_phase * np.pi)
            Phi_jam_mat = np.asmatrix(np.diag(jam_full), dtype=complex)

            # 有源放大: β × Φ (参考 ris_functions-jin: beta[i] = sqrt(1+10*rand))
            # 信号路径: β × Φ_signal → 放大用户信号
            # 干扰路径: β × Φ_jam → 放大干扰噪声
            beta_signal = np.asmatrix(np.diag(np.full(reflect_elements, ris_beta)), dtype=complex)
            beta_jam = np.asmatrix(np.diag(np.full(jam_elements, ris_beta)), dtype=complex)

            # 构建完整放大矩阵 (64×64对角)
            amp_full = np.zeros((self.RIS.ant_num, self.RIS.ant_num), dtype=complex)
            amp_full[:reflect_elements, :reflect_elements] = np.asarray(beta_signal)
            amp_full[reflect_elements:reflect_elements + jam_elements,
            reflect_elements:reflect_elements + jam_elements] = np.asarray(beta_jam)
            amp_diag = np.asmatrix(amp_full, dtype=complex)

            # 应用放大: θ = β × Φ
            self.RIS.Phi_signal = amp_diag @ Phi_signal_mat  # 信号: 放大给用户
            self.RIS.Phi_jam = amp_diag @ Phi_jam_mat  # 干扰: 放大干扰窃听者
            self.RIS.Phi = self.RIS.Phi_signal

            # RIS功率约束 (参考 FrisModule: limit_power)
            # 总功率 = 反射功率 + 干扰功率 + 热噪声 ≤ P_RIS_MAX
            P_RIS_MAX_mW = 10 ** (self.RIS.Pr / 10)  # 使用RIS配置的功率约束
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            # 使用活跃端口的信道
            active_port = self.UAV_FAS.fas_active_port
            h_UR_active = h_UR[:, active_port:active_port + 1]  # (N_ris, 1)

            # 反射功率: ||θ_R @ h_UR_active||² × F²
            P_reflect = 0
            Phi_R = np.asarray(self.RIS.Phi_signal)
            P_reflect = np.linalg.norm(Phi_R @ h_UR_active) ** 2 * self.UAV_FAS.F ** 2

            # 干扰功率: ||θ_J @ h_UR_active||² × F²
            P_jam = 0
            Phi_J = np.asarray(self.RIS.Phi_jam)
            P_jam = np.linalg.norm(Phi_J @ h_UR_active) ** 2 * self.UAV_FAS.F ** 2

            # 热噪声功率
            P_noise = (np.linalg.norm(Phi_R, 'fro') ** 2 + np.linalg.norm(Phi_J, 'fro') ** 2) * dB_to_normal(
                -114) * 1e-3

            P_RIS_total = P_reflect + P_jam + P_noise

            if P_RIS_total > P_RIS_MAX_mW:
                scale = np.sqrt(P_RIS_MAX_mW / (P_RIS_total + 1e-10))
                self.RIS.Phi_signal *= scale
                self.RIS.Phi_jam *= scale
                self.RIS.Phi = self.RIS.Phi_signal

            # 记录实际放大倍数
            self.RIS.amplification_gains = np.full(self.RIS.ant_num, ris_beta)

        # 干扰功率模型: Phi_jam已包含放大增益β，an_power归一化为1.0
        # 干扰等效噪声 = |h_R_p @ Phi_jam @ h_UR|² × 1.0
        # 其中Phi_jam = β × Φ_jam_mat，放大增益已体现在Phi_jam中
        self.an_power = 1.0  # 归一化，避免与β重复计算
        self.z_AN = (np.random.randn(self.RIS.ant_num, 1) + 1j * np.random.randn(self.RIS.ant_num, 1)) / np.sqrt(2)

        # 更新信道容量
        self.update_channel_capacity()

        # 存储系统状态
        self.store_current_system_sate()

        # 获取新状态
        new_state = self.observe()

        # 获取奖励 (已在reward()中包含能耗惩罚和边界检查)
        reward = self.reward()

        # 边界检查: 出界时施加渐进惩罚
        done = False
        x, y = self.UAV_FAS.coordinate[0:2]
        border_penalty = 0
        if x < self.border[0][0]:
            border_penalty += abs(x - self.border[0][0]) * 0.1
        elif x > self.border[0][1]:
            border_penalty += abs(x - self.border[0][1]) * 0.1
        if y < self.border[1][0]:
            border_penalty += abs(y - self.border[1][0]) * 0.1
        elif y > self.border[1][1]:
            border_penalty += abs(y - self.border[1][1]) * 0.1

        if border_penalty > 0:
            reward -= border_penalty  # 渐进惩罚，不直接终止
            done = border_penalty > 5.0  # 超出边界5m才终止

        self.data_manager.store_data([reward], 'reward')

        # 存储RIS相位数据
        if self.if_with_FAS:
            # RIS信号反射相位 (归一化到[-1,1])
            signal_phase = np.angle(np.diag(np.asarray(self.RIS.Phi_signal))).tolist()
            self.data_manager.store_data([np.mean(signal_phase) / math.pi], 'RIS_signal_phase')
            # RIS干扰相位 (归一化到[-1,1])
            jam_phase = np.angle(np.diag(np.asarray(self.RIS.Phi_jam))).tolist()
            self.data_manager.store_data([np.mean(jam_phase) / math.pi], 'RIS_jam_phase')
            # FAS活跃端口号
            self.data_manager.store_data([self.UAV_FAS.fas_active_port], 'FAS_active_port')

        return new_state, reward, done, []

    def reward(self):
        """计算奖励

        设计原则:
        1. Agent 1学习: FAS端口选择、RIS放大增益、RIS干扰比例、RIS信号/干扰相位
        2. Agent 2学习: UAV轨迹 (vx, vy)
        3. 奖励聚焦: 保密速率 + FAS辅助安全 + 空间引导 + 约束惩罚
        """
        import math

        # === 1. 保密速率 SSR (主目标) ===
        total_secrecy = 0
        user_secrecies = []
        for user in self.user_list:
            secrecy_k = max(0, user.capacity - max(self.eavesdrop_capacity_array[:, user.index]))
            total_secrecy += secrecy_k
            user_secrecies.append(secrecy_k)

        # === 2. FAS端口保密增益 ===
        # 衡量端口选择: 用户信道增益 vs 窃听者信道增益
        R_fas = 0
        active_port = self.UAV_FAS.fas_active_port
        if len(self.attacker_list) > 0:
            fas_risks = []
            for user in self.user_list:
                g_user = abs(np.asarray(self.h_U_k[user.index].channel_matrix)[active_port, 0]) ** 2
                g_eve = abs(np.asarray(self.h_U_p[self.attacker_list[0].index].channel_matrix)[active_port, 0]) ** 2
                if g_user > 0:
                    fas_risks.append(max(0, (g_user - g_eve) / (g_user + 1e-10)))
            if fas_risks:
                R_fas = np.mean(fas_risks)

        # === 3. 空间位置奖励: 引导UAV飞向安全加权中点 ===
        user_positions = np.array([u.coordinate[:2] for u in self.user_list])
        attacker_positions = np.array([a.coordinate[:2] for a in self.attacker_list])
        uav_pos = np.array(self.UAV_FAS.coordinate[:2])

        # 计算安全加权中点：离窃听者越远的用户，权重越大
        if len(attacker_positions) > 0:
            attacker_pos = attacker_positions[0]  # 第一个窃听者
            dist_to_attacker = np.array([np.linalg.norm(user_pos - attacker_pos) for user_pos in user_positions])
            weights = dist_to_attacker / (np.sum(dist_to_attacker) + 1e-10)  # 归一化权重
            target_pos = np.sum(user_positions * weights[:, np.newaxis], axis=0)  # 加权平均
        else:
            target_pos = np.mean(user_positions, axis=0)  # 无窃听者时飞向中点

        dist_to_target = np.linalg.norm(uav_pos - target_pos)
        # 空间引导: 距离目标越近奖励越高，最大1.0
        # 使用指数衰减替代线性，近距离梯度更大
        R_spatial = max(0, 1 - dist_to_target / 50)  # [0, 1]

        # === 4. 功率约束惩罚 ===
        # F²是线性功率增益，P_actual = F² × P_base_mW
        # K个端口等功率分配，总功率 = F_total² = F² × K
        P_base_mW = 1.0  # 基础发射功率 (mW)
        K = self.num_active_ports
        P_actual_mW = self.UAV_FAS.F ** 2 * K * P_base_mW  # 实际总功率 (mW)
        P_max_mW = 10 ** (self.UAV_FAS.F_Pmax / 10)  # 1000 mW
        p_m = max(0, P_actual_mW - P_max_mW) / (P_max_mW + 1e-10)

        # === 5. 最低安全速率惩罚 ===
        R_th = 0.03  # log2下阈值调大 (原0.01 × 3.32)
        p_r = 0
        for sk in user_secrecies:
            if sk < R_th:
                p_r += (R_th - sk) / R_th

        # === 6. 能耗惩罚 ===
        v_t = getattr(self.UAV_FAS, 'v_t', 0)
        E_p = get_energy_consumption(v_t)
        E_p_norm = (E_p - ENERGY_MIN) / (ENERGY_MAX - ENERGY_MIN + 1e-10)
        E_p_norm = max(0, min(1, E_p_norm))
        lambda_e = 0.03
        p_e = lambda_e * E_p_norm

        # === 7. 窃听者容量惩罚 (最坏情况) ===
        # 取所有用户中窃听容量的最大值（最坏情况）
        max_eavesdrop = np.max(self.eavesdrop_capacity_array)
        # 归一化到[0,1]: 以5bits/s/Hz为参考上限
        p_eve = min(max_eavesdrop / 15.0, 1.0)  # log2下参考值调大 (原5.0 × 3.32)
        # 自适应惩罚系数：窃听容量越大，惩罚越重
        lambda_eve = 0.5 + 1.0 * min(max_eavesdrop / 3.0, 1.0)  # 范围 [0.5, 1.5]

        # === 8. RIS干扰相位对齐奖励 ===
        # 鼓励Agent将RIS干扰相位对齐窃听者信道，最大化干扰效果
        R_ris_jam = 0
        if len(self.attacker_list) > 0 and self.if_with_FAS:
            try:
                active_port = self.UAV_FAS.fas_active_port
                h_R_p = np.asarray(self.h_R_p[0].channel_matrix).flatten()  # (N_ris,)
                h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
                # 窃听者等效信道相位
                eve_phase = np.angle(h_R_p * h_UR[:, active_port])
                # RIS干扰相位
                jam_diag = np.diag(np.asarray(self.RIS.Phi_jam))
                jam_phase = np.angle(jam_diag)
                # 计算对齐度: 相位差越小，对齐度越高
                phase_diff = np.abs(jam_phase - eve_phase)
                # 归一化到[0,1]: 完美对齐=1, 完全相反=0
                alignment = np.mean(np.cos(phase_diff))  # [-1, 1] → 映射到[0, 1]
                R_ris_jam = (alignment + 1.0) / 2.0
            except:
                R_ris_jam = 0

        # === 组合奖励 ===
        raw_reward = (0.12 * total_secrecy  # 主目标: 保密速率
                      + 0.1 * R_fas  # FAS辅助安全
                      + 0.25 * R_spatial  # 空间引导
                      + 0.15 * R_ris_jam  # RIS干扰相位对齐奖励
                      - 0.1 * p_m  # 功率约束
                      - 0.5 * p_r  # 最低安全速率约束
                      - 0.05 * p_e  # 能耗惩罚
                      - lambda_eve * p_eve)  # 窃听者容量惩罚

        return np.clip(raw_reward, -5.0, 5.0)

    def find_optimal_uav_position(self, grid_step=10):
        """
        网格搜索最优UAV位置，最大化总安全容量
        返回: (最优x, 最优y, 最大安全容量)
        """
        best_pos = None
        best_sec = -float('inf')

        # 保存原始位置
        original_pos = self.UAV_FAS.coordinate.copy()

        for x in range(self.border[0][0], self.border[0][1] + 1, grid_step):
            for y in range(self.border[1][0], self.border[1][1] + 1, grid_step):
                # 设置临时位置
                self.UAV_FAS.coordinate[0] = x
                self.UAV_FAS.coordinate[1] = y

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
        self.UAV_FAS.coordinate = original_pos

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
            UAV_position_list = list(self.UAV_FAS.coordinate)

        # 系统状态信息 (14维)
        system_state = [
            self.UAV_FAS.fas_num_ports / 16.0,  # FAS端口数归一化
            self.RIS.ant_num / 64.0,  # RIS单元数归一化
            self.user_num / 4.0,  # 用户数归一化
            self.attacker_num / 2.0,  # 窃听者数归一化
            2.2 / 4.0,  # αur归一化
            2.2 / 4.0,  # αul归一化 (纯LoS)
            2.8 / 4.0,  # αr归一化
            (sigma_n + 120) / 40.0,  # 噪声功率归一化
            P_max_dBm / 40.0,  # 最大功率归一化
            D_max / 1.0,  # 最大移动距离归一化
            delta_t / 1.0,  # 时隙时长归一化
            self.UAV_FAS.coordinate[2] / 100.0,  # UAV高度归一化
            self.UAV_FAS.fas_active_port / 12.0,  # 当前端口归一化
            0.0,  # 保留
        ]

        return port_channel_list + UAV_position_list + system_state

    def store_current_system_sate(self):
        """存储当前系统状态"""
        # 存储FAS信息（活跃端口列表和增益）
        active_ports = getattr(self.UAV_FAS, 'fas_active_ports', [self.UAV_FAS.fas_active_port])
        row_data = [active_ports, self.UAV_FAS.F]
        self.data_manager.store_data(row_data, 'beamforming_matrix')

        # 存储RIS反射相位矩阵
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1, -1)))[0, :])
        self.data_manager.store_data(row_data, 'reflecting_coefficient')

        # 存储 UAV 状态
        row_data = list(self.UAV_FAS.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')

        # 存储功率信息
        row_data = [self.UAV_FAS.F ** 2, self.UAV_FAS.F_Pmax]
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

    def _ris_reflected_channel(self, h_R, ris_phi, amp_gains=None):
        """
        计算有源RIS反射路径 (参考ris_functions-jin-xinhaofngda.py)
        h_R (1, N_ris) @ diag(beta_i) @ Phi (N_ris, N_ris) @ h_UR.T (N_ris, N_FAS) → (1, N_FAS)

        有源RIS: 每个反射单元有放大增益 beta_i
        amp_gains: 放大增益对角矩阵 (N_ris, N_ris)，None时使用self.RIS.Phi中的增益
        """
        h_R_a = np.asarray(h_R)  # (1, N_ris)
        Phi_a = np.asarray(ris_phi)  # (N_ris, N_ris)
        h_UR_a = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)

        # 有源RIS放大增益: 使用step()中设置的统一增益
        if amp_gains is None:
            amp_gains = np.diag(self.RIS.amplification_gains)  # (N_ris, N_ris)

        # 有源反射: h_R @ diag(beta) @ Phi @ h_UR
        return h_R_a @ amp_gains @ Phi_a @ h_UR_a  # (1, N_FAS)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """计算攻击者p的综合信道 (标量，基于多个活跃端口)
        注意: Phi_signal 已包含 β 放大，不再额外乘 amp_gains
        """
        active_ports = getattr(self.UAV_FAS, 'fas_active_ports', [self.UAV_FAS.fas_active_port])
        if self.if_with_FAS:
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_R_p = np.asarray(self.h_R_p[p].channel_matrix)  # (1, N_ris)
            Phi_signal = np.asarray(self.RIS.Phi_signal)  # 已含β
            H_eff = 0
            for port in active_ports:
                h_d = np.asarray(self.h_U_p[p].channel_matrix)[port, 0]  # 标量
                h_UR_active = h_UR[:, port]  # (N_ris,)
                H_reflect = h_R_p @ Phi_signal @ h_UR_active  # 标量 (不再乘amp_gains)
                H_eff += h_d + H_reflect
            return np.array([[H_eff]])  # (1, 1)
        else:
            return np.asarray(self.h_U_p[p].channel_matrix)[active_ports[0]:active_ports[0] + 1, :]

    def calculate_comprehensive_channel_of_user_k(self, k):
        """计算用户k的综合信道 (标量，基于多个活跃端口)
        注意: Phi_signal 已包含 β 放大，不再额外乘 amp_gains
        """
        active_ports = getattr(self.UAV_FAS, 'fas_active_ports', [self.UAV_FAS.fas_active_port])
        if self.if_with_FAS:
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # (1, N_ris)
            Phi_signal = np.asarray(self.RIS.Phi_signal)  # 已含β
            H_eff = 0
            for port in active_ports:
                h_d = np.asarray(self.h_U_k[k].channel_matrix)[port, 0]  # 标量
                h_UR_active = h_UR[:, port]  # (N_ris,)
                H_reflect = h_R_k @ Phi_signal @ h_UR_active  # 标量 (不再乘amp_gains)
                H_eff += h_d + H_reflect
            return np.array([[H_eff]])  # (1, 1)
        else:
            return np.asarray(self.h_U_k[k].channel_matrix)[active_ports[0]:active_ports[0] + 1, :]

    def calculate_capacity_of_user_k(self, k):
        """计算用户k的信道容量（用户级波束成形）

        信号模型: 用户k收到的信号 = Σ_i w_k[i] × (h_d_i + H_reflect_i) × F × s_k
        其中 w_k[i] 是用户k在端口i上的波束成形权重

        修复: 加入用户专属相位偏移，让不同用户的RIS信道自然差异化
        """
        noise_power = self.user_list[k].noise_power
        active_ports = getattr(self.UAV_FAS, 'fas_active_ports', [self.UAV_FAS.fas_active_port])
        w_k = self.user_beamforming_weights[k]  # (num_active_ports,) 用户k的波束权重

        if self.if_with_FAS:
            h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # (1, N_ris)
            Phi_signal = np.asarray(self.RIS.Phi_signal)

            # 用户专属相位偏移: 不同用户在RIS处有不同的入射角
            # 相位偏移 = 2π × k / K，确保每个用户的信道有独特特征
            user_phase_offset = np.exp(1j * np.pi * k / self.user_num)
            h_R_k_modified = h_R_k * user_phase_offset  # 应用用户专属相位

            # 功率合并（非相干）: 每个端口独立计算功率，按权重加权求和
            power_k = 0
            for i, port in enumerate(active_ports):
                h_d = np.asarray(self.h_U_k[k].channel_matrix)[port, 0]
                h_UR_active = h_UR[:, port]
                H_reflect = h_R_k_modified @ Phi_signal @ h_UR_active  # 使用修改后的信道
                h_eff_i = (h_d + H_reflect) * self.UAV_FAS.F
                power_k += w_k[i] * abs(h_eff_i) ** 2
            alpha_k = power_k
        else:
            alpha_k = abs(np.asarray(self.h_U_k[k].channel_matrix)[active_ports[0], 0] * self.UAV_FAS.F) ** 2

        beta_k = dB_to_normal(noise_power) * 1e-3
        return math.log2(1 + alpha_k / (beta_k + 1e-20))

    def calculate_capacity_array_of_attacker_p(self, p):
        """计算攻击者p对各用户的窃听容量（按用户分别计算）

        窃听模型: 窃听者截获用户k的信号，容量取决于:
        1. 窃听者信道 (h_U_p) - 相同
        2. RIS反射到窃听者的信道 (h_R_p) - 相同
        3. 用户k的专属相位偏移 - 不同! 影响窃听者解码能力
        4. 波束权重 (w_k) - 不同

        修复: 加入用户专属相位偏移，让不同用户的窃听容量自然差异化
        """
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        active_ports = getattr(self.UAV_FAS, 'fas_active_ports', [self.UAV_FAS.fas_active_port])

        # 预计算窃听者侧的信道 (与k无关)
        h_UR = np.asarray(self.h_UR.channel_matrix).T  # (N_ris, N_FAS)
        h_R_p = np.asarray(self.h_R_p[p].channel_matrix)  # (1, N_ris)
        Phi_signal = np.asarray(self.RIS.Phi_signal)  # 已含β
        Phi_jam = np.asarray(self.RIS.Phi_jam)  # 已含β

        # RIS干扰功率 (与k无关)
        jam_power = 0
        for port in active_ports:
            h_UR_active = h_UR[:, port]
            jam_ch = h_R_p @ Phi_jam @ h_UR_active
            jam_power += abs(jam_ch) ** 2

        # 计算每个用户的窃听容量
        caps = []
        for k in range(K):
            w_k = self.user_beamforming_weights[k]

            # 用户专属相位偏移: 与用户容量计算保持一致
            user_phase_offset = np.exp(1j * np.pi * k / self.user_num)
            h_R_k = np.asarray(self.h_R_k[k].channel_matrix)  # 用户k的RIS信道
            h_R_k_modified = h_R_k * user_phase_offset  # 应用用户专属相位

            # 窃听者截获用户k的信号: 使用用户k的专属RIS信道
            h_eff_power_k = []
            for port in active_ports:
                h_d = np.asarray(self.h_U_p[p].channel_matrix)[port, 0]
                h_UR_active = h_UR[:, port]
                channel_k = h_d + h_R_k_modified @ Phi_signal @ h_UR_active
                h_eff_power_k.append(abs(channel_k * self.UAV_FAS.F) ** 2)

            # 分子: 用户k的信号功率 (w_k加权 + 用户专属信道)
            signal_power_k = sum(w_k[i] * h_eff_power_k[i] for i in range(len(active_ports)))

            # 分母: 噪声 + RIS干扰 (无多用户干扰)
            beta_p = dB_to_normal(noise_power) * 1e-3 + jam_power

            caps.append(math.log2(1 + signal_power_k / (beta_p + 1e-20)))

        return np.array(caps)

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
        43维 = 12维端口选择 + 1维F增益 + 1维RIS放大增益 + 1维RIS干扰比例
               + 24维RIS相位 + 4维用户级波束成形权重(K×num_active_ports)
        """
        result = 0
        if self.if_with_FAS:
            result += self.UAV_FAS.fas_num_ports  # 端口选择: 12维 (Gumbel-Softmax)
            result += 1  # FAS增益 F: 1维
            result += 1  # RIS放大增益 β: 1维
            result += 1  # RIS干扰比例 η ∈ [0,1]: 1维 ← Agent控制
            result += 2 * 12  # RIS相位: 24维 (12信号+12干扰)
            result += self.user_num * self.num_active_ports  # 用户级波束权重: K×num_active_ports
        return result

    def get_system_state_dim(self):
        """获取状态维度 (Agent 1)
        89维 = 各端口信道 + UAV位置(3) + 系统状态(14)
        包含：12端口×3实体×2(实虚) + UAV位置 + 系统参数
        """
        result = 0
        # 各端口到用户和窃听者的信道（实部+虚部）
        result += 2 * (self.user_num + self.attacker_num) * self.UAV_FAS.fas_num_ports
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
        state.extend(list(self.UAV_FAS.coordinate))

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