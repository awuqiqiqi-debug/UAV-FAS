# UAV-BS-FAS: 流体天线辅助无人机保密通信系统

基于深度强化学习的 **UAV-BS-FAS（无人机基站-流体天线系统）** 保密通信联合优化。

## 系统模型

在存在窃听者的场景下，通过联合优化 **UAV飞行轨迹、主动波束赋形、FAS天线端口选择和RIS被动波束赋形**，最大化保密能量效率（SEE）。

```
    ┌─────────────────────────────────────────────┐
    │              UAV-BS-FAS (无人机)              │
    │  ┌──────┐  ┌──────┐                         │
    │  │  BS  │  │ FAS  │  12个流体天线端口         │
    │  │ 4天线 │  │      │  灵活端口选择增强空间分集   │
    │  └──┬───┘  └──┬───┘                         │
    └─────┼─────────┼─────────────────────────────┘
          │    ↓    │
     直射链路    RIS反射链路
          │    ↓    │
    ┌─────┼────┼────┼─────┐
    │  ┌──┴──┐│ ┌──┴──┐  │
    │  │用户1 ││ │用户2 │  │  ← 合法用户 (相长干涉增强)
    │  └─────┘│ └─────┘  │
    │         │ ┌───────┐│
    │         │ │窃听者  ││  ← 窃听者 (相消干涉抑制)
    │         │ └───────┘│
    └─────────┼──────────┘
          ┌───┴───┐
          │  RIS  │  64个反射单元 (8×8阵列)
          │地面固定│  被动波束赋形
          └───────┘
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 载波频率 | 28 GHz | 毫米波频段 |
| BS天线数 | 4 | 无人机基站天线 |
| FAS端口数 | 12 | 流体天线端口 |
| RIS单元数 | 64 | 8×8反射阵列 |
| 最大发射功率 | 30 dBm | 无人机功率上限 |
| 噪声功率 | -114 dBm | 接收端高斯噪声 |
| 时隙长度 | 0.1 s | 单个优化时隙 |
| 最大飞行距离 | 0.25 m/时隙 | 无人机移动约束 |
| RIS位置 | (0, 50, 12.5) m | 地面固定 |

## 算法框架

采用 **双DRL框架**，两个智能体分别负责：

1. **Agent 1 (波束赋形)** → 联合优化 UAV 主动波束赋形 + RIS 被动波束赋形 + FAS 端口选择
2. **Agent 2 (轨迹)** → 优化 UAV 飞行轨迹

| 算法 | 说明 |
|------|------|
| **Twin-TD3** | 双延迟深度确定性策略梯度（核心算法） |
| **DDPG** | 深度确定性策略梯度（对比基线） |
| **Active RIS** | 有源RIS增强版（`train_aris.py`） |

## 优化目标

- **SSR (Sum Secrecy Rate)** — 最大化保密速率之和
- **SEE (Secrecy Energy Efficiency)** — 最大化保密能量效率（保密速率/能耗）

## 环境安装

```bash
conda create --name uav-fas python=3.10
conda activate uav-fas
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
# Twin-TD3 + SEE（保密能量效率）
python main_train_uav_bs_fas.py --drl td3 --reward see

# Twin-TD3 + SSR（保密速率）
python main_train_uav_bs_fas.py --drl td3 --reward ssr

# DDPG 对比基线
python main_train_uav_bs_fas.py --drl ddpg --reward see

# 指定训练轮数（默认350轮）
python main_train_uav_bs_fas.py --drl td3 --reward see --ep-num 500

# 从已有检查点继续训练
python main_train_uav_bs_fas.py --drl td3 --reward see --load-path data/storage/scratch/<DIR>

# Active RIS 增强版训练
python train_aris.py
```

### 评估与可视化

```shell
# 批量评估
bash batch_eval.sh

# 绘制轨迹、速率、能量效率图
python load_and_plot.py --path data/storage/scratch/<DIR> --ep-num 350
```

## 能耗模型

采用经典旋翼无人机功率消耗模型：

```
P = P₀ + Pᵢ + P_tip + P_body
```

| 参数 | 值 | 说明 |
|------|-----|------|
| P₀ | 580.65 W | 桨叶基准剖面功率 |
| Pᵢ | 790.67 W | 旋翼诱导功率 |
| U_tip | 200 m/s | 桨尖线速度 |
| m | 1.3 kg | 无人机质量 |
| ρ | 1.225 kg/m³ | 空气密度 |
| A_r | 0.79 m² | 旋翼桨盘面积 |

## 项目结构

```
Twin-TD3-main/
├── main_train_uav_bs_fas.py   # UAV-BS-FAS 主训练脚本
├── train_aris.py              # Active RIS 训练
├── env_uav_bs_fas.py          # UAV-BS-FAS 环境模型
├── td3.py                     # Twin-TD3 智能体
├── ddpg.py                    # DDPG 智能体
├── channel.py                 # 毫米波信道建模 (UMi)
├── entity.py                  # 实体定义 (UAV, 用户, 窃听者, RIS)
├── math_tool.py               # 数学工具
├── render.py                  # 可视化
├── data_manager.py            # 数据管理
├── generate_plots*.py         # 绘图工具
├── batch_train.sh / batch_eval.sh
└── requirements.txt
```

## 参考文献

本项目基于以下研究工作：

- **RIS 信道仿真**: [SimRIS Channel Simulator](https://ieeexplore.ieee.org/document/9282349) — [Python 实现](https://github.com/Brook1711/RIS_components)
- **系统模型**: [RIS-aided mmWave UAV Communications](https://doi.org/10.1109/LWC.2021.3081464) — [代码](https://github.com/Brook1711/WCL-pulish-code)
- **旋翼无人机能耗**: [Energy Minimization in IoT Based on Rotary-Wing UAV](https://doi.org/10.1109/LWC.2019.2916549)
- **TD3 算法**: [PyTorch TD3](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/TD3)
- **原始框架**: [Twin-TD3 (yjwong1999)](https://github.com/yjwong1999/Twin-TD3)
