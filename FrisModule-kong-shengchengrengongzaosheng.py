import numpy as np
import math

class FrisModule:
    """
    多功能RIS模块 - 实现反射、干扰和动态元素调度功能
    
    功能特性:
    1. 信号反射: 将基站信号反射到目标用户
    2. 干扰生成: 生成人工噪声干扰窃听者
    3. 动态调度: 根据场景需求分配RIS元素用途
    
    工作模式:
    - MR (Multi-functional RIS): 同时反射和干扰
    - AR (Active RIS): 仅反射
    - AJ (Active Jamming): 仅干扰
    """
    
    def __init__(self, ant_num=16, Pr=0, P_J=0, beta=10, sigma=-100):
        """
        初始化RIS模块
        
        参数:
            ant_num: RIS反射单元数量
            Pr: RIS总功率 (dBm)
            P_J: 干扰功率 (dBm)
            beta: 反射单元幅度 (dB)
            sigma: 热噪声功率 (dBm)
        """
        self.ant_num = ant_num
        self.Pr = Pr          # 总功率约束 (dBm)
        self.P_J = P_J        # 干扰功率 (dBm)
        self.beta = beta      # 反射单元幅度
        self.sigma = sigma    # 热噪声功率
        
        # 相位矩阵初始化
        self.theta_R = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)  # 反射相位矩阵
        self.theta_J = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex) # 干扰相位矩阵
        self.theta = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)    # 总相位矩阵
        
        # 记录数据
        self.P_RIS = []       # RIS消耗功率记录
        self.betas = []       # 幅度记录
        self.skds = []        # 调度记录
        self.skd_rate = 1     # 调度率
    
    def calculate_channel_alignment_theta(self, H_br, h_rk, h_bk):
        """
        计算信道对齐相位偏移
        
        参数:
            H_br: BS到RIS的信道矩阵
            h_rk: RIS到用户/攻击者的信道矩阵
            h_bk: BS到用户/攻击者的直传信道矩阵
        
        返回:
            theta: 优化后的相位偏移向量
        """
        theta = (np.diag(np.array(h_rk).flatten()) @ np.array(H_br)).flatten()
        phase_diff = np.angle(np.array(h_bk).flatten()) - np.angle(theta)
        return np.exp(1j * phase_diff.flatten())
    
    def configure_phase_shift(self, action, UA_theta, if_with_jamming=True, if_with_reflect=True):
        """
        配置RIS相位偏移和元素调度
        
        参数:
            action: 强化学习动作向量 [-1, 1]
            UA_theta: 用户和攻击者的信道对齐相位列表
            if_with_jamming: 是否启用干扰功能
            if_with_reflect: 是否启用反射功能
        
        返回:
            Phi: 综合相位偏移向量
            skd: 反射单元选择矩阵
            rskd: 干扰单元选择矩阵
        """
        theta_skd = (action[self.ant_num:self.ant_num * 2] + 1) / 2
        Phi = np.zeros(self.ant_num, dtype=complex)
        rskd = 0
        
        if if_with_jamming and if_with_reflect:  # MR模式：同时反射和干扰
            UA_num = len(UA_theta)
            for i in range(UA_num):
                temp = np.where((theta_skd >= i / UA_num) & (theta_skd < (i + 1) / UA_num), 1, 0)
                Phi += UA_theta[i] @ np.diag(temp)
                if i == UA_num - 1:
                    rskd = temp  # 最后一个分配给干扰
            skd = 1 - rskd
            skd = np.asmatrix(np.diag(skd))
            rskd = np.asmatrix(np.diag(rskd))
        
        elif not if_with_jamming and if_with_reflect:  # AR模式：仅反射
            A_num = len(UA_theta) - 1 if if_with_jamming else len(UA_theta)
            for i in range(A_num):
                temp = np.where((theta_skd >= i / A_num) & (theta_skd < (i + 1) / A_num), 1, 0)
                Phi += UA_theta[i] @ np.diag(temp)
            skd = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
            rskd = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex)
        
        elif if_with_jamming and not if_with_reflect:  # AJ模式：仅干扰
            Phi = UA_theta[-1]
            skd = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex)
            rskd = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
        
        return Phi, skd, rskd
    
    def update_phase_matrices(self, action, UA_theta, if_with_jamming=True, if_with_reflect=True):
        """
        更新RIS相位矩阵
        
        参数:
            action: 强化学习动作向量 [-1, 1]
            UA_theta: 用户和攻击者的信道对齐相位列表
            if_with_jamming: 是否启用干扰功能
            if_with_reflect: 是否启用反射功能
        
        返回:
            state_phase_shift: 当前相位状态
        """
        # 获取幅度参数
        beta = (action[:self.ant_num] + 1) * np.sqrt(self.beta) / 2
        self.betas.append(beta)
        beta = np.asmatrix(np.diag(beta))
        
        # 计算相位偏移和元素调度
        Phi, skd, rskd = self.configure_phase_shift(action, UA_theta, if_with_jamming, if_with_reflect)
        state_phase_shift = Phi
        Phi = np.asmatrix(np.diag(Phi))
        
        # 更新调度记录
        self.skds.append(np.sum(np.diagonal(skd) == 1))
        self.skd_rate = float(np.sum(np.diagonal(skd) == 1) / self.ant_num)
        
        # 更新相位矩阵
        self.theta_J = beta * Phi * rskd  # 干扰相位矩阵
        self.theta = beta * Phi           # 总相位矩阵
        self.theta_R = beta * Phi * skd   # 反射相位矩阵
        
        return state_phase_shift
    
    def calculate_comprehensive_channel(self, H_br, h_rk, h_bk):
        """
        计算综合信道（反射路径 + 直传路径）
        
        参数:
            H_br: BS到RIS的信道矩阵
            h_rk: RIS到用户的信道矩阵
            h_bk: BS到用户的直传信道矩阵
        
        返回:
            comprehensive_channel: 综合信道矩阵
        """
        return h_rk * self.theta_R * H_br + h_bk
    
    def calculate_jamming_power(self, H_br, h_rp, G):
        """
        计算对窃听者的干扰功率
        
        参数:
            H_br: BS到RIS的信道矩阵
            h_rp: RIS到窃听者的信道矩阵
            G: 基站波束成形矩阵
        
        返回:
            jamming: 干扰功率
        """
        jamming = 0
        user_num = G.shape[1]
        for z in range(user_num):
            jamming += math.pow(np.linalg.norm(h_rp * self.theta_J * H_br * G[:, z]), 2)
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
    
    def reset(self):
        """
        重置RIS状态
        """
        self.theta_R = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
        self.theta_J = np.asmatrix(np.diag(np.zeros(self.ant_num, dtype=complex)), dtype=complex)
        self.theta = np.asmatrix(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
        self.skd_rate = 1
        self.P_RIS = []
        self.betas = []
        self.skds = []
    
    @staticmethod
    def _dbm_to_mw(dbm):
        """
        dBm转mW
        """
        return 10 ** (dbm / 10)
    
    @staticmethod
    def _mw_to_dbm(mw):
        """
        mW转dBm
        """
        return 10 * math.log10(mw)


# ================ 使用示例 ================
if __name__ == '__main__':
    # 初始化RIS模块
    ris = FrisModule(ant_num=16, Pr=0, P_J=10, beta=10)
    
    # 模拟信道矩阵
    H_br = np.matrix(np.random.randn(16, 4) + 1j * np.random.randn(16, 4))
    h_rk = np.matrix(np.random.randn(1, 16) + 1j * np.random.randn(1, 16))
    h_bk = np.matrix(np.random.randn(1, 4) + 1j * np.random.randn(1, 4))
    G = np.matrix(np.random.randn(4, 2) + 1j * np.random.randn(4, 2))
    
    # 计算信道对齐相位
    theta_user = ris.calculate_channel_alignment_theta(H_br, h_rk, h_bk)
    theta_attacker = ris.calculate_channel_alignment_theta(H_br, h_rk, h_bk)  # 模拟攻击者
    
    # 模拟强化学习动作
    action = np.random.uniform(-1, 1, 34)  # 16(幅度) + 16(调度) + 2(预留)
    
    # 更新相位矩阵 (MR模式)
    UA_theta = [theta_user, theta_attacker]
    state_phase = ris.update_phase_matrices(action, UA_theta, if_with_jamming=True, if_with_reflect=True)
    
    # 计算综合信道
    comp_channel = ris.calculate_comprehensive_channel(H_br, h_rk, h_bk)
    
    # 计算干扰功率
    jamming = ris.calculate_jamming_power(H_br, h_rk, G)
    
    # 计算并限制功率
    power = ris.limit_power(H_br, G)
    
    print(f"RIS模块初始化完成")
    print(f"相位状态维度: {state_phase.shape}")
    print(f"综合信道维度: {comp_channel.shape}")
    print(f"干扰功率: {jamming:.4f} mW")
    print(f"RIS消耗功率: {power:.4f} mW")
    print(f"调度率: {ris.skd_rate:.2f}")
