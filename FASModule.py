import numpy as np


class UAVMountedFAS:
    """
    无人机搭载 波束可重构离散端口流体天线 FAS
    适用模型：UAV-FAS发射信号→RIS反射→合法用户；同时发射人工噪声干扰窃听者
    核心原理：有限离散空间端口切换，改变波束指向，实现对准RIS、零陷窃听者
    """

    def __init__(self, num_ports=8):
        # FAS 端口数量 常用8/16端口
        self.num_ports = num_ports

        # 定义无人机机身上 FAS 离散端口坐标 (局部坐标系)
        # 水平线性排布，模拟液态金属微流控端口阵列
        # 28GHz: λ=c/f≈1.071cm, 端口间距=0.5λ≈0.00536m
        self.port_spacing = 0.5 * 3e8 / 28e9  # 0.5λ = 0.00536m
        self.port_local = np.array([
            [self.port_spacing * i, 0.0, 0.0] for i in range(-self.num_ports // 2, self.num_ports // 2)
        ])

    def get_fas_abs_position(self, uav_global_pos, port_idx):
        """
        输入：无人机全局坐标 + 选中端口索引
        输出：FAS当前辐射单元 全局绝对坐标
        """
        if port_idx < 0 or port_idx >= self.num_ports:
            raise ValueError("端口索引超出FAS端口范围")
        # 局部坐标映射到全局坐标
        fas_abs_pos = uav_global_pos + self.port_local[port_idx]
        return fas_abs_pos

    def path_loss(self, tx_pos, rx_pos, alpha=2.0):
        """路径损耗模型"""
        dist = np.linalg.norm(tx_pos - rx_pos)
        return 1.0 / (dist ** alpha)

    def get_channel_gain(self, uav_pos, port_idx, target_pos):
        """
        计算：UAV-FAS 到 目标(RIS/用户/窃听者) 的信道增益
        """
        fas_pos = self.get_fas_abs_position(uav_pos, port_idx)
        gain = self.path_loss(fas_pos, target_pos)
        return gain, fas_pos

    def fas_beam_optimize(self, uav_pos, ris_pos, eaves_pos):
        """
        遍历所有FAS端口，自动选最优端口：
        最大化 FAS→RIS 增益，同时最小化 FAS→窃听者 增益
        实现：对准RIS传输 + 天然零陷干扰窃听者
        """
        best_port = 0
        best_metric = -np.inf
        best_gain_ris = 0
        best_gain_eve = 0

        for idx in range(self.num_ports):
            g_ris, _ = self.get_channel_gain(uav_pos, idx, ris_pos)
            g_eve, _ = self.get_channel_gain(uav_pos, idx, eaves_pos)
            # 优化目标：RIS增益 - 窃听者增益
            metric = g_ris - g_eve

            if metric > best_metric:
                best_metric = metric
                best_port = idx
                best_gain_ris = g_ris
                best_gain_eve = g_eve

        return best_port, best_gain_ris, best_gain_eve