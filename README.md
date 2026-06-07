# UAV-FAS-BS: 流体天线辅助无人机保密通信

基于深度强化学习的 **UAV-BS-FAS（无人机基站-流体天线系统）** 保密通信联合优化。

## 项目概述

本项目研究 **物理层安全** 问题：UAV基站搭载流体天线系统（FAS）和RIS，服务合法用户的同时抑制窃听者。

系统通过联合优化 **UAV飞行轨迹、主动波束赋形、FAS天线端口选择和RIS被动波束赋形**，最大化保密能量效率（SEE）。

### 已实现算法

| 算法 | 说明 |
|------|------|
| **Twin-TD3** | 双延迟深度确定性策略梯度（核心算法） |
| **DDPG** | 深度确定性策略梯度（对比基线） |
| **PPO** | 近端策略优化 |
| **SAC** | 软演员-评论家 |

## Project Structure

```
.
├── Twin-TD3-main/              # Core algorithm implementation (based on TTD3)
│   ├── td3.py                  # Twin-TD3 agent
│   ├── ddpg.py                 # DDPG agent
│   ├── env_uav_bs_fas.py       # UAV-BS-FAS environment
│   ├── env.py                  # Base environment
│   ├── env1.py                 # Environment variant
│   ├── entity.py               # Entity definitions (UAV, users, eavesdroppers)
│   ├── channel.py              # Channel modeling (FAS + RIS)
│   ├── math_tool.py            # Mathematical utilities
│   ├── data_manager.py         # Data management
│   ├── render.py               # Visualization
│   ├── main_train.py           # Main training script
│   ├── main_train_uav_bs_fas.py # FAS-specific training
│   ├── train_aris.py           # Active RIS training
│   ├── batch_train.sh          # Batch training script
│   ├── batch_eval.sh           # Batch evaluation script
│   ├── generate_plots*.py      # Plotting utilities
│   └── requirements.txt        # Python dependencies
├── PPO.py                      # PPO agent implementation
├── SAC.py                      # SAC agent implementation
├── td3.py                      # TD3 agent (root)
├── FASModule.py                # Fluid Antenna System module
├── ris_functions-jin-xinhaofngda.py # RIS functions
├── FrisModule-kong-shengchengrengongzaosheng.py # FRISS module
├── env.py                      # Environment (root)
├── entity.py                   # Entity definitions (root)
├── channel.py                  # Channel modeling (root)
├── math_tool.py                # Math utilities (root)
├── data_manager.py             # Data manager (root)
├── render.py                   # Visualization (root)
├── main_PPO.py                 # PPO training entry
├── main_SAC.py                 # SAC training entry
├── main_td3.py                 # TD3 training entry
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/awuqiqiqi-debug/UAV-FAS-BS.git
cd UAV-FAS-BS

# Create conda environment
conda create --name uav-fas python=3.10
conda activate uav-fas

# Install dependencies
pip install -r Twin-TD3-main/requirements.txt
```

### Training

```bash
# Twin-TD3 training (UAV-BS-FAS scenario)
cd Twin-TD3-main
python main_train_uav_bs_fas.py

# Active RIS training
python train_aris.py

# Batch training
bash batch_train.sh
```

### Evaluation

```bash
# Batch evaluation
bash batch_eval.sh

# Generate plots
python generate_plots.py
```

## 系统模型

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

| 参数 | 值 |
|------|-----|
| 载波频率 | 28 GHz (毫米波) |
| BS天线 | 4 |
| FAS端口 | 12 |
| RIS单元 | 64 (8×8) |
| 发射功率上限 | 30 dBm |
| 噪声功率 | -114 dBm |

## Acknowledgement

This project builds upon the [Twin-TD3](https://github.com/yjwong1999/Twin-TD3) framework for RIS-aided UAV communication. RIS simulation is based on the [SimRIS Channel Simulator](https://github.com/Brook1711/RIS_components).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
