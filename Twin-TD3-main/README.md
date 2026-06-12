# UAV-FAS: 流体天线辅助无人机保密通信系统

基于深度强化学习的 **UAV-FAS（无人机-流体天线系统）** 保密通信联合优化。

## 系统模型

在存在窃听者的场景下，通过联合优化 **UAV飞行轨迹、FAS天线端口选择和有源RIS波束赋形**，最大化保密速率（SSR）。

```
    ┌─────────────────────────────────────┐
    │           UAV-FAS (无人机)           │
    │  ┌──────────────────────────────┐  │
    │  │  FAS 流体天线 (12端口)        │  │  FAS 作为唯一发射天线
    │  │  端口切换 + 增益控制          │  │  灵活端口选择增强空间分集
    │  └──────────────┬───────────────┘  │
    └─────────────────┼──────────────────┘
                      │
           ┌──────────┼──────────┐
           │ 直射链路  │ RIS反射链路│
           ▼          ▼          ▼
      ┌────────┐  ┌────────┐  ┌────────┐
      │ 用户1  │  │  RIS   │  │ 用户2  │  ← 合法用户 (相长干涉增强)
      │单天线  │  │64单元  │  │单天线  │
      └────────┘  │有源放大│  └────────┘
                  │+人工噪声│
                  └────┬───┘
                       │ 干扰链路
                       ▼
                  ┌──────────┐
                  │ 窃听者1  │  ← 窃听者 (干扰抑制)
                  │ 单天线   │
                  └──────────┘
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 载波频率 | 28 GHz | 毫米波频段 |
| FAS端口数 | 12 | 流体天线端口（唯一发射天线） |
| RIS单元数 | 64 | 8×8有源反射阵列 |
| 最大发射功率 | 30 dBm | 无人机功率上限 |
| 噪声功率 | -114 dBm | 接收端高斯噪声 |
| 时隙长度 | 0.1 s | 单个优化时隙 |
| 最大水平速度 | 1.0 m/s | 无人机水平移动速度上限 |
| 固定飞行高度 | 50 m | 高度固定，仅水平移动 |
| RIS位置 | (0, 50, 12.5) m | 地面固定 |
| 飞行边界 | X:[-50,50], Y:[-50,50] m | 水平空域范围 |

## 算法框架

采用 **双DRL框架**，两个智能体分别负责：

1. **Agent 1 (FAS+RIS)** → 联合优化 FAS 端口选择 + FAS增益 + 有源RIS放大增益 + RIS相位
2. **Agent 2 (轨迹)** → 优化 UAV 水平飞行轨迹（vx, vy）

| 算法 | 说明 |
|------|------|
| **Twin-TD3** | 双延迟深度确定性策略梯度（核心算法） |
| **DDPG** | 深度确定性策略梯度（对比基线） |

## 优化目标

- **SSR (Sum Secrecy Rate)** — 最大化保密速率之和
- **奖励函数**: 基于论文 Eq.(11) 的 tanh 归一化设计

## 环境安装

```bash
conda create --name uav-fas python=3.10
conda activate uav-fas
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
# Twin-TD3 + SSR（默认2000轮）
python src/main_train.py --drl td3 --reward ssr

# Twin-TD3 + SEE（保密能量效率）
python src/main_train.py --drl td3 --reward see

# DDPG 对比基线
python src/main_train.py --drl ddpg --reward ssr

# 指定训练轮数
python src/main_train.py --drl td3 --reward ssr --ep-num 500

# 从已有检查点继续训练
python src/main_train.py --drl td3 --reward ssr --load-path data/storage/scratch/<DIR>
```

### 评估与可视化

```bash
# 绘制轨迹、速率、能量效率图
python scripts/load_and_plot.py --path data/storage/scratch/<DIR> --ep-num 2000

# 生成实验结果图表
python scripts/generate_plots.py --path data/storage/scratch/<DIR>
```

## 动作空间

### Agent 1 (FAS + RIS) — 38 维

| 参数范围 | 维度 | 含义 |
|---------|------|------|
| [0:12] | 12 | FAS 端口选择 (softmax logits) |
| [12:13] | 1 | FAS 增益 F ∈ [1, 3] |
| [13:14] | 1 | RIS 放大增益 β ∈ [1, √11] |
| [14:38] | 24 | RIS 相位 (12信号 + 12干扰) |

### Agent 2 (UAV 轨迹) — 2 维

| 参数范围 | 维度 | 含义 |
|---------|------|------|
| [0:1] | 1 | vx 水平速度 ∈ [-1, 1] |
| [1:2] | 1 | vy 水平速度 ∈ [-1, 1] |

## 状态空间

### Agent 1 — 89 维

| 组成部分 | 维度 |
|---------|------|
| 各端口到用户/窃听者的信道 (实+虚) | 2×3×12 = 72 |
| UAV 位置坐标 | 3 |
| 系统状态信息 | 14 |

### Agent 2 — 18 维

| 组成部分 | 维度 |
|---------|------|
| UAV 坐标 | 3 |
| 用户位置 (×2) | 6 |
| RIS 位置 | 3 |
| 窃听者位置 | 3 |
| 用户信道容量 (×2) | 2 |
| 窃听者信道容量 | 1 |

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
├── src/                        # 核心代码
│   ├── agents/                 # DRL 智能体
│   │   ├── td3_agent.py        # Twin-TD3 智能体
│   │   └── ddpg_agent.py       # DDPG 智能体
│   ├── networks/               # 神经网络结构
│   │   └── actor_critic.py     # Actor-Critic 网络定义
│   ├── envs/                   # 环境模型
│   │   ├── entity.py           # 实体定义 (UAV, RIS, User, Attacker)
│   │   ├── channel.py          # 毫米波 LoS 信道模型
│   │   ├── math_tool.py        # 数学工具函数
│   │   ├── uav_comm_env.py     # 主环境 (UAV-FAS)
│   │   ├── uav_comm_env_legacy.py  # 旧版环境
│   │   └── minimal_irs_env.py  # 最小 IRS 环境
│   ├── utils/                  # 工具模块
│   │   ├── data_manager.py     # 数据管理 (读写 .mat/.xlsx)
│   │   └── renderer.py         # 3D 可视化
│   ├── tests/                  # 测试
│   │   └── test_uav_comm.py    # 环境单元测试
│   └── main_train.py           # 主训练脚本入口
├── scripts/                    # 辅助脚本
│   ├── generate_plots.py       # 统一绘图工具 (--path 参数化)
│   ├── load_and_plot.py        # 训练结果分析
│   ├── run_simulation.py       # 仿真运行
│   ├── run_and_report.py       # 自动报告生成
│   ├── batch_train.sh          # 批量训练
│   ├── batch_eval.sh           # 批量评估
│   └── legacy/                 # 旧版训练脚本归档
├── configs/
│   └── config.yaml             # 配置文件
├── runs/                       # 训练产出 (git 忽略)
├── data/
│   └── init_location.xlsx      # 实体初始位置
├── docs/                       # 文档
├── .gitignore
├── requirements.txt
└── README.md
```

## 参考文献

本项目基于以下研究工作：

- **RIS 信道仿真**: [SimRIS Channel Simulator](https://ieeexplore.ieee.org/document/9282349) — [Python 实现](https://github.com/Brook1711/RIS_components)
- **系统模型**: [RIS-aided mmWave UAV Communications](https://doi.org/10.1109/LWC.2021.3081464) — [代码](https://github.com/Brook1711/WCL-pulish-code)
- **旋翼无人机能耗**: [Energy Minimization in IoT Based on Rotary-Wing UAV](https://doi.org/10.1109/LWC.2019.2916549)
- **TD3 算法**: [PyTorch TD3](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/TD3)
- **原始框架**: [Twin-TD3 (yjwong1999)](https://github.com/yjwong1999/Twin-TD3)
