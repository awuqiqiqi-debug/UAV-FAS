import numpy as np
import math

class UAV(object):
    """
    UAV object with coordinate 
    And with ULA antenas, default 8 
    And limited power
    And with fixed rotation angle
    """
    def __init__(self, coordinate, index = 0, rotation = 0, ant_num=16, ant_type = 'ULA', max_movement_per_time_slot = 0.5):
        """
        coordinate is the init coordinate of UAV, meters, np.array
        """
        self.max_movement_per_time_slot = max_movement_per_time_slot
        self.type = 'UAV'
        self.coordinate = coordinate
        self.rotation = rotation
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
        self.index = index

        # init beamforming matrix in UAV (must be inited in env.py)
        self.G = np.asmatrix(np.zeros((ant_num, 1)))
        self.G_Pmax = 0

    def reset(self, coordinate):
        """
        reset UAV coordinate
        """
        self.coordinate = coordinate
        
    def update_coor_sys(self, delta_angle):
        """
        used in function move to update the relevant coordinate system 
        """
        self.rotation = self.rotation + delta_angle
        coor_sys_x = np.array([\
        math.cos(self.rotation),\
        math.sin(self.rotation),\
        0])
        coor_sys_z = np.array([\
        0,\
        0,\
        -1])
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = np.array([coor_sys_x,coor_sys_y,coor_sys_z])
        
    def update_coordinate(self, distance_delta_d, direction_fai):
        """
        used in function move to update UAV cordinate
        """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai, delta_angle = 0):
        """
        preform the 2D movement every step
        """
        self.update_coordinate(distance_delta_d, direction_fai)
        self.update_coor_sys(delta_angle)

class RIS(object):
    """
    多功能可重构智能反射面 (Multi-functional RIS):
    1. 信号反射：将基站信号反射到目标用户，支持幅度放大
    2. 干扰生成：生成人工噪声干扰窃听者
    3. 动态调度：根据场景需求分配RIS元素用途

    工作模式:
    - MR (Multi-functional RIS): 同时反射和干扰
    - AR (Active RIS): 仅反射（带放大）
    - AJ (Active Jamming): 仅干扰
    """
    def __init__(self, coordinate, coor_sys_z, index = 0, ant_num=4, ant_type = 'UPA',
                 Pr=30, P_J=20, beta=10, sigma=-100):
        """
        coordinate: RIS位置坐标, meters, np.array
        coor_sys_z: RIS法向量
        ant_num: RIS反射单元数量
        Pr: RIS总功率约束 (dBm) - 有源RIS放大功率 (默认30dBm=1W)
        P_J: 干扰功率 (dBm) (默认20dBm=100mW)
        beta: 反射单元幅度增益 (dB) - 有源放大增益
        sigma: 热噪声功率 (dBm)
        """
        self.type = 'RIS'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        coor_sys_z = coor_sys_z / np.linalg.norm(coor_sys_z)
        coor_sys_x = np.cross(coor_sys_z, np.array([0,0,1]))
        coor_sys_x = coor_sys_x / np.linalg.norm(coor_sys_x)
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = [coor_sys_x,coor_sys_y,coor_sys_z]
        self.index = index

        # 多功能有源RIS参数
        self.Pr = Pr          # 总功率约束 (dBm) - 有源RIS放大功率
        self.P_J = P_J        # 干扰功率 (dBm)
        self.beta = beta      # 反射单元幅度增益 (dB) - 有源放大增益
        self.sigma = sigma    # 热噪声功率 (dBm)

        # 有源RIS放大增益 (参考ris_functions-jin-xinhaofngda.py)
        # beta_i = sqrt(1 + 10*rand) - 每个元件的放大增益
        self.amplification_gains = np.sqrt(1 + 10 * np.random.rand(self.ant_num))

        # 相位矩阵初始化
        self.Phi = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)  # 总相位矩阵
        self.theta_R = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)  # 反射相位矩阵
        self.theta_J = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex)  # 干扰相位矩阵

        # 记录数据
        self.P_RIS = []       # RIS消耗功率记录
        self.betas = []       # 幅度记录
        self.skds = []        # 调度记录
        self.skd_rate = 1     # 调度率（反射单元占比）

    def calculate_channel_alignment_theta(self, H_br, h_rk, h_bk):
        """
        计算信道对齐相位偏移

        参数:
            H_br: BS到RIS的信道矩阵, shape (RIS_ant_num, bs_ant_num)
            h_rk: RIS到用户的信道向量, shape (1, RIS_ant_num) 或 (RIS_ant_num, 1)
            h_bk: BS到用户的直传信道向量, shape (1, bs_ant_num) 或 (bs_ant_num, 1)

        返回:
            theta: 优化后的相位偏移向量, shape (RIS_ant_num,)
        """
        h_rk_flat = np.array(h_rk).flatten()  # (RIS_ant_num,)
        h_bk_flat = np.array(h_bk).flatten()  # (bs_ant_num,)
        H_br_arr = np.array(H_br)  # (RIS_ant_num, bs_ant_num)

        # 对每个RIS元素，计算与直传信道的相位对齐
        theta = np.zeros(self.ant_num, dtype=complex)
        for n in range(self.ant_num):
            # RIS元素n通过BS天线到用户的等效信道
            effective_channel = h_rk_flat[n] * H_br_arr[n, :]  # (bs_ant_num,)
            # 与直传信道的相位差
            phase_diff = np.angle(h_bk_flat) - np.angle(effective_channel)
            # 最优相位偏移（对BS天线取平均）
            theta[n] = np.exp(1j * np.mean(phase_diff))

        return theta

    def calculate_joint_phase_optimization(self, H_br, h_rk_list, h_bk_list, h_rp_list, alpha_cancel=0.0):
        """
        模块1：联合反射相位优化 — 4元素有限自由度下，全部用于增强用户信号
        窃听者抑制依靠信道差异 + 干扰实现

        参数:
            H_br: BS到RIS信道, shape (ant_num, bs_ant_num)
            h_rk_list: RIS到用户信道列表
            h_bk_list: BS到用户直传信道列表
            h_rp_list: RIS到窃听者信道列表
            alpha_cancel: 窃听者抵消权重 (0=纯用户增强)

        返回:
            theta_reflect: 反射相位向量, shape (ant_num,)
            theta_jam: 干扰相位向量, shape (ant_num,)
        """
        N = self.ant_num
        H_br_arr = np.array(H_br)
        K = len(h_rk_list)
        theta_reflect = np.zeros(N, dtype=complex)
        theta_jam = np.zeros(N, dtype=complex)

        for n in range(N):
            # === 反射相位：逐元素独立优化 ===
            # 每个RIS元素n选择与其信道增益最强的BS天线对齐
            user_sum = np.zeros(1, dtype=complex)
            for k in range(K):
                h_rk = np.array(h_rk_list[k]).flatten()
                h_bk = np.array(h_bk_list[k]).flatten()
                bs_ant_num = H_br_arr.shape[1]
                best_phase = 0
                best_gain = 0
                for m in range(bs_ant_num):
                    # BS天线m通过RIS元素n到用户k的信道增益
                    gain = abs(h_rk[n] * H_br_arr[n, m])
                    # 反射路径相位 = angle(h_rk[n] * H_br[n,m])
                    reflect_ph = np.angle(h_rk[n] * H_br_arr[n, m])
                    # 直传路径相位 = angle(h_bk[m])
                    direct_ph = np.angle(h_bk[m])
                    # 对齐目标：反射+直传同相 → theta = direct_ph - reflect_ph
                    phase_target = direct_ph - reflect_ph
                    if gain > best_gain:
                        best_gain = gain
                        best_phase = phase_target
                user_sum += np.exp(1j * best_phase)

            # 纯用户增强，不做窃听者抵消（4元素自由度不够）
            theta_reflect[n] = np.angle(user_sum)

            # === 干扰相位：100%瞄准窃听者 ===
            h_rp = np.array(h_rp_list[0]).flatten()
            attacker_phase = np.angle(np.sum(h_rp[n] * H_br_arr[n, :]))
            theta_jam[n] = -attacker_phase  # 对齐窃听者信道，最大化噪声

        return theta_reflect, theta_jam

    def calculate_jamming_beamforming(self, H_br, h_rk_list, h_rp_list):
        """
        模块2：RIS干扰波束 — 纯瞄准窃听者（不做用户零陷，4元素不够）
        """
        N = self.ant_num
        H_br_arr = np.array(H_br)
        jam_phases = np.zeros(N, dtype=complex)

        for n in range(N):
            h_rp = np.array(h_rp_list[0]).flatten()
            eve_phase = np.angle(np.sum(h_rp[n] * H_br_arr[n, :]))
            jam_phases[n] = -eve_phase  # 纯瞄准窃听者

        return jam_phases

    def calculate_adaptive_jamming_power(self, h_rp, h_rk_list, P_J_base_mw, P_J_max_mw):
        """
        模块2补充：自适应干扰功率控制
        窃听者信道强时加大干扰，信道弱时降低干扰

        参数:
            h_rp: RIS到窃听者信道, shape (1, ant_num)
            h_rk_list: RIS到用户信道列表
            P_J_base_mw: 基础干扰功率 (mW)
            P_J_max_mw: 最大干扰功率 (mW)

        返回:
            P_J_adaptive_mw: 自适应干扰功率 (mW)
        """
        h_rp_arr = np.array(h_rp).flatten()
        eve_gain = np.sum(np.abs(h_rp_arr) ** 2)
        avg_user_gain = np.mean([np.sum(np.abs(np.array(h).flatten()) ** 2) for h in h_rk_list])
        ratio = eve_gain / (avg_user_gain + 1e-10)
        jam_scale = min(1.0, 0.3 + 0.7 * min(1.0, ratio / 2.0))
        return P_J_base_mw + jam_scale * (P_J_max_mw - P_J_base_mw)

    def generate_jamming_phases(self):
        """
        生成随机干扰相位（非相干干扰）
        干扰相位不应与任何信道对齐，而应产生随机噪声效果
        这样干扰信号在窃听者处是非相干叠加，降低其SINR
        """
        random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, self.ant_num))
        return random_phases

    def configure_phase_shift(self, action, UA_theta, if_with_jamming=True, if_with_reflect=True,
                              heuristic_reflect=None, heuristic_jam=None):
        """
        配置RIS相位偏移和元素调度

        参数:
            action: 强化学习动作向量 [-1, 1]
            UA_theta: 用户和攻击者的信道对齐相位列表
            if_with_jamming: 是否启用干扰功能
            if_with_reflect: 是否启用反射功能
            heuristic_reflect: 启发式反射相位, shape (ant_num,), 可选
            heuristic_jam: 启发式干扰相位, shape (ant_num,), 可选

        返回:
            Phi: 综合相位偏移向量
            skd: 反射单元选择矩阵
            rskd: 干扰单元选择矩阵
        """
        theta_skd = (action[self.ant_num:self.ant_num * 2] + 1) / 2
        Phi = np.zeros(self.ant_num, dtype=complex)
        rskd = 0

        if if_with_jamming and if_with_reflect:  # MR模式
            if heuristic_reflect is not None:
                # Phi=反射相位, skd=全1（全部元素可反射）
                Phi = heuristic_reflect.copy()
                skd = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
                # rskd: 全部元素也可用于干扰（干扰相位由heuristic_jam提供）
                rskd = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
            else:
                # 旧模式：RL调度元素分配
                UA_num = len(UA_theta)
                for i in range(UA_num):
                    temp = np.where((theta_skd >= i / UA_num) & (theta_skd < (i + 1) / UA_num), 1, 0)
                    if i == UA_num - 1:
                        Phi += UA_theta[i] @ np.diag(temp)
                        rskd = temp
                    else:
                        Phi += UA_theta[i] @ np.diag(temp)
                skd = 1 - rskd
                skd = np.asmatrix(np.diag(skd))
                rskd = np.asmatrix(np.diag(rskd))

        elif not if_with_jamming and if_with_reflect:  # AR模式：仅反射
            A_num = len(UA_theta)
            for i in range(A_num):
                temp = np.where((theta_skd >= i / A_num) & (theta_skd < (i + 1) / A_num), 1, 0)
                if heuristic_reflect is not None:
                    Phi += heuristic_reflect @ np.diag(temp)
                else:
                    Phi += UA_theta[i] @ np.diag(temp)
            skd = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
            rskd = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex)

        elif if_with_jamming and not if_with_reflect:  # AJ模式：仅干扰
            if heuristic_jam is not None:
                Phi = heuristic_jam
            else:
                Phi = UA_theta[-1]
            skd = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex)
            rskd = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)

        return Phi, skd, rskd

    def update_phase_matrices(self, action, UA_theta, if_with_jamming=True, if_with_reflect=True,
                              heuristic_reflect=None, heuristic_jam=None):
        """
        更新RIS相位矩阵

        参数:
            action: 强化学习动作向量 [-1, 1]
            UA_theta: 用户和攻击者的信道对齐相位列表
            if_with_jamming: 是否启用干扰功能
            if_with_reflect: 是否启用反射功能
            heuristic_reflect: 启发式反射相位, shape (ant_num,), 可选
            heuristic_jam: 启发式干扰相位, shape (ant_num,), 可选

        返回:
            state_phase_shift: 当前相位状态
        """
        # 获取幅度参数
        beta = (action[:self.ant_num] + 1) * np.sqrt(self.beta) / 2
        self.betas.append(beta)
        beta = np.asmatrix(np.diag(beta))

        # 计算相位偏移和元素调度
        Phi, skd, rskd = self.configure_phase_shift(action, UA_theta, if_with_jamming, if_with_reflect,
                                                     heuristic_reflect=heuristic_reflect, heuristic_jam=heuristic_jam)
        state_phase_shift = Phi
        Phi = np.asmatrix(np.diag(Phi))

        # 更新调度记录
        self.skds.append(np.sum(np.diagonal(skd) == 1))
        self.skd_rate = float(np.sum(np.diagonal(skd) == 1) / self.ant_num)

        # 更新相位矩阵
        if heuristic_jam is not None:
            self.theta_J = beta * np.asmatrix(np.diag(heuristic_jam), dtype=complex)  # 干扰相位矩阵
        else:
            self.theta_J = beta * Phi * rskd  # 干扰相位矩阵
        self.theta = beta * Phi           # 总相位矩阵
        self.theta_R = beta * Phi * skd   # 反射相位矩阵
        self.Phi = Phi

        return state_phase_shift

    def calculate_comprehensive_channel(self, H_br, h_rk, h_bk):
        """
        计算综合信道（反射路径 + 直传路径）

        参数:
            H_br: BS到RIS的信道矩阵
            h_rk: RIS到用户的信道向量
            h_bk: BS到用户的直传信道向量

        返回:
            comprehensive_channel: 综合信道矩阵
        """
        return h_rk * self.theta_R * H_br + h_bk

    def calculate_jamming_power(self, H_br=None, h_rp=None, G=None, h_FR=None, F=None):
        """
        计算对窃听者的干扰功率（信号级人工噪声模型）

        RIS干扰原理：theta_J对RIS接收的信号施加相位扰动，产生指向窃听者的人工噪声。
        干扰功率 = alpha_J * ||h_R_p^H * theta_J||^2
        其中 alpha_J 是干扰比例因子，表示RIS将多少比例的能量转为干扰。

        返回:
            jamming: 干扰噪声功率 (归一化单位)
        """
        if h_rp is not None:
            h_rp_H = h_rp.H  # (1, RIS_ant_num)
            # 信号级干扰：RIS对窃听者方向产生人工噪声
            jamming = math.pow(np.linalg.norm(h_rp_H * self.theta_J), 2)
        else:
            jamming = math.pow(np.linalg.norm(self.theta_J, 'fro'), 2)
        return jamming

    def calculate_ris_power_consumption(self, H_br, G):
        """
        计算RIS消耗的总功率

        参数:
            H_br: BS到RIS的信道矩阵
            G: 基站波束成形矩阵

        返回:
            P_RIS: RIS消耗的总功率 (mW)
        """
        P_RIS = 0
        user_num = G.shape[1]

        # 反射功率
        for i in range(user_num):
            P_RIS += math.pow(np.linalg.norm(self.theta_R * H_br * G[:, i]), 2)

        # 热噪声功率
        P_RIS += math.pow(np.linalg.norm(self.theta, 'fro'), 2) * self._dbm_to_mw(self.sigma)

        # 干扰功率
        P_RIS += math.pow(np.linalg.norm(self.theta_J, 'fro'), 2) * self._dbm_to_mw(self.P_J)

        return P_RIS

    def limit_power(self, H_br, G):
        """
        限制RIS功率不超过最大约束

        参数:
            H_br: BS到RIS的信道矩阵
            G: 基站波束成形矩阵

        返回:
            P_RIS: 受限后的功率 (mW)
        """
        P_RIS = self.calculate_ris_power_consumption(H_br, G)

        if P_RIS > self._dbm_to_mw(self.Pr):
            scaleFactor = np.sqrt(self._dbm_to_mw(self.Pr) / P_RIS)
            self.theta_J *= scaleFactor
            self.theta *= scaleFactor
            self.theta_R *= scaleFactor
            P_RIS = self.calculate_ris_power_consumption(H_br, G)

        self.P_RIS.append(P_RIS)
        return P_RIS

    @staticmethod
    def _dbm_to_mw(dbm):
        """dBm转mW"""
        return 10 ** (dbm / 10)

    @staticmethod
    def _mw_to_dbm(mw):
        """mW转dBm"""
        return 10 * math.log10(mw)

class UAV_FAS(object):
    """
    UAV-FAS Entity (无人机-流体天线一体化平台):
    - UAV mobility platform (无人机移动平台)
    - FAS: 流体天线系统作为唯一发射天线

    无人机搭载设备：
    - FAS流体天线：12个离散端口(ULA)，通过端口切换改变波束指向
    - RL直接选择最优端口，F为标量增益

    核心原理：RL从12个端口中选择最优端口，实现对准用户、零陷窃听者
    """
    def __init__(self, coordinate, index = 0, rotation = 0,
                 fas_num_ports=12, fas_ant_type='ULA',
                 max_movement_per_time_slot = 0.5):
        """
        coordinate: UAV initial coordinate in meters, np.array
        fas_num_ports: number of FAS discrete ports (离散端口数)
        """
        self.max_movement_per_time_slot = max_movement_per_time_slot
        self.type = 'UAV'  # 信道模型匹配用（channel.py 中 'UAV_user'/'UAV_RIS' 等）
        self.coordinate = coordinate
        self.rotation = rotation
        self.index = index

        # FAS parameters (流体天线系统 — 唯一发射天线)
        self.fas_num_ports = fas_num_ports
        self.ant_num = fas_num_ports  # For compatibility with channel model
        self.fas_ant_type = fas_ant_type
        self.ant_type = fas_ant_type  # For compatibility with channel model
        # 28GHz: λ=c/f≈1.071cm, 端口间距=0.5λ≈0.00536m
        self.fas_port_spacing = 0.5 * 3e8 / 28e9  # 0.5λ = 0.00536m
        self.fas_port_local = np.array([
            [self.fas_port_spacing * i, 0.0, 0.0] for i in range(-fas_num_ports // 2, fas_num_ports // 2)
        ])  # 离散端口局部坐标（水平线性排布）
        self.fas_active_port = 0  # 当前激活的端口索引（由RL选择）
        self.F = 1.0  # FAS增益（标量，由RL控制）
        self.F_Pmax = 0

        # Coordinate system
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]

    def reset(self, coordinate):
        """Reset UAV coordinate"""
        self.coordinate = coordinate
        self.fas_active_port = 0
        self.F = 1.0

    def update_coor_sys(self, delta_angle):
        """Update coordinate system"""
        self.rotation = self.rotation + delta_angle
        coor_sys_x = np.array([\
        math.cos(self.rotation),\
        math.sin(self.rotation),\
        0])
        coor_sys_z = np.array([\
        0,\
        0,\
        -1])
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = np.array([coor_sys_x,coor_sys_y,coor_sys_z])

    def update_coordinate(self, distance_delta_d, direction_fai):
        """Update coordinate"""
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai, delta_angle = 0):
        """Perform 2D movement"""
        self.update_coordinate(distance_delta_d, direction_fai)
        self.update_coor_sys(delta_angle)

class User(object):
    """
    user with single antenna
    """
    def __init__(self, coordinate, index, ant_num = 1, ant_type = 'single'):
        """
        coordinate is the init coordinate of user, meters, np.array
        ant_num is the antenna number of user
        """
        self.type = 'user'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # init the capacity
        self.capacity = 0
        self.secure_capacity = 0
        self.QoS_constrain = 0
        # init the comprehensive_channel
        self.comprehensive_channel = 0
        # init receive noise sigma in dB
        self.noise_power = -114

    def reset(self, coordinate):
        """
        reset user coordinate
        """
        self.coordinate = coordinate
        
    def update_coordinate(self, distance_delta_d, direction_fai):
        """
        used in function move to update UAV cordinate
        """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai):
        """
        preform the 2D movement every step
        """
        self.update_coordinate(distance_delta_d, direction_fai)
        
class Attacker(object):
    """
    Attacker with single antenna
    """
    def __init__(self, coordinate, index, ant_num = 1, ant_type= 'single'):
        """
        coordinate is the init coordinate of Attacker, meters, np.array
        ant_num is the antenna number of Attacker
        """
        self.type = 'attacker'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # init the capacity, this is a K length np.array
        self.capacity = 0
        self.comprehensive_channel = 0
        # init receive noise sigma in dBmW
        self.noise_power = -114

    def reset(self, coordinate):
        """
        reset attacker coordinate
        """
        self.coordinate = coordinate

    def update_coordinate(self, distance_delta_d, direction_fai):
        """
        used in function move to update UAV cordinate
        """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai):
        """
        preform the 2D movement every step
        """
        self.update_coordinate(distance_delta_d, direction_fai)