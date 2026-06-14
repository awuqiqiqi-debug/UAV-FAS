# UAV-FAS: 流体天线辅助无人机保密通信

基于深度强化学习的 **UAV-FAS（无人机-流体天线系统）** 保密通信联合优化。

## 项目概述

本项目研究 **物理层安全** 问题：UAV搭载流体天线系统（FAS）和有源RIS，服务合法用户的同时抑制窃听者。

系统通过联合优化 **UAV飞行轨迹、FAS天线端口选择和有源RIS波束赋形**，最大化保密速率（SSR）。

### 已实现算法

| 算法 | 说明 |
|------|------|
| **Twin-TD3** | 双延迟深度确定性策略梯度（核心算法） |
| **DDPG** | 深度确定性策略梯度（对比基线） |
| **SAC** | 软演员-评论家（最大熵RL，自动温度调节） |

## 项目结构

```
.
├── Twin-TD3-main/                  # 核心算法实现
│   ├── src/                        # 核心代码
│   │   ├── agents/                 # DRL 智能体
│   │   │   ├── td3_agent.py        # Twin-TD3 智能体
│   │   │   ├── ddpg_agent.py       # DDPG 智能体
│   │   │   └── sac_agent.py        # SAC 智能体
│   │   ├── networks/               # 神经网络结构
│   │   │   └── actor_critic.py     # Actor-Critic 网络定义
│   │   ├── envs/                   # 环境模型
│   │   │   ├── entity.py           # 实体定义 (UAV, RIS, User, Attacker)
│   │   │   ├── channel.py          # 毫米波 LoS 信道模型
│   │   │   ├── math_tool.py        # 数学工具函数
│   │   │   ├── uav_comm_env.py     # 主环境 (UAV-FAS)
│   │   │   ├── uav_comm_env_legacy.py  # 旧版环境
│   │   │   └── minimal_irs_env.py  # 最小 IRS 环境
│   │   ├── utils/                  # 工具模块
│   │   │   ├── data_manager.py     # 数据管理
│   │   │   └── renderer.py         # 3D 可视化
│   │   ├── tests/                  # 测试
│   │   │   └── test_uav_comm.py    # 环境单元测试
│   │   ├── main_train.py           # TD3 主训练脚本
│   │   ├── main_train_sac.py       # SAC 训练脚本
│   │   └── main_train_td3_velocity.py  # 速度约束 TD3 训练
│   ├── scripts/                    # 辅助脚本
│   │   ├── generate_plots.py       # 统一绘图工具
│   │   ├── load_and_plot.py        # 训练结果分析
│   │   ├── run_simulation.py       # 仿真运行
│   │   ├── batch_train.sh          # 批量训练
│   │   ├── batch_eval.sh           # 批量评估
│   │   └── legacy/                 # 旧版训练脚本归档
│   ├── configs/                    # 配置文件
│   ├── runs/                       # 训练产出 (git 忽略)
│   ├── data/                       # 数据文件
│   ├── docs/                       # 文档
│   └── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 快速开始

### 环境安装

```bash
conda create --name uav-fas python=3.10
conda activate uav-fas
pip install -r Twin-TD3-main/requirements.txt
```

### 训练

```bash
cd Twin-TD3-main

# Twin-TD3 + SSR（默认2000轮）
python src/main_train.py --drl td3 --reward ssr

# DDPG 对比基线
python src/main_train.py --drl ddpg --reward ssr

# SAC 训练（最大熵RL，天然探索）
python src/main_train_sac.py --reward ssr --ep-num 2000

# 速度约束 TD3（Actor输出速度vx/vy，环境内转换为位移）
python src/main_train_td3_velocity.py --reward ssr --ep-num 2000

# 从检查点继续训练
python src/main_train.py --drl td3 --reward ssr --load-path data/storage/scratch/<DIR>
```

### 评估

```bash
cd Twin-TD3-main

# 绘制训练结果图
python scripts/load_and_plot.py --path data/storage/scratch/<DIR>

# 生成实验图表
python scripts/generate_plots.py --path data/storage/scratch/<DIR>
```

## 系统模型

```
    ┌─────────────────────────────────────┐
    │           UAV-FAS (无人机)           │
    │  ┌──────────────────────────────┐  │
    │  │  FAS 流体天线 (12端口)        │  │  FAS 作为唯一发射天线
    │  │  端口切换 + 增益控制          │  │
    │  └──────────────┬───────────────┘  │
    └─────────────────┼──────────────────┘
                      │
           ┌──────────┼──────────┐
           │ 直射链路  │ RIS反射链路│
           ▼          ▼          ▼
      ┌────────┐  ┌────────┐  ┌────────┐
      │ 用户1  │  │  RIS   │  │ 用户2  │
      │单天线  │  │64单元  │  │单天线  │
      └────────┘  │有源放大│  └────────┘
                  │+人工噪声│
                  └────┬───┘
                       ▼
                  ┌──────────┐
                  │ 窃听者1  │
                  └──────────┘
```

| 参数 | 值 |
|------|-----|
| 载波频率 | 28 GHz (毫米波) |
| FAS端口 | 12（唯一发射天线，Gumbel Top-K，同时激活2~3端口） |
| RIS单元 | 64 (8×8，有源放大，Agent控制干扰比例) |
| RIS位置 | (20, -20, 12.5) m（靠近窃听者） |
| 最大水平速度 | 1.0 m/s |
| 固定飞行高度 | 50 m |
| 发射功率上限 | 30 dBm |
| 噪声功率 | -114 dBm |

## 致谢

本项目基于 [Twin-TD3](https://github.com/yjwong1999/Twin-TD3) 框架。RIS 仿真基于 [SimRIS Channel Simulator](https://github.com/Brook1711/RIS_components)。

## 许可证

本项目基于 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。
