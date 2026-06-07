# UAV-FAS-BS: Fluid Antenna System Aided UAV Secure Communication

Reinforcement learning-based secure communication optimization for **Fluid Antenna System (FAS)** and **Reconfigurable Intelligent Surface (RIS)** assisted UAV base station scenarios.

## Project Overview

This project investigates **physical layer security** in UAV communication systems, where a UAV base station serves legitimate users while suppressing eavesdroppers. The system incorporates:

- **Fluid Antenna System (FAS)** — flexible antenna placement for enhanced signal control
- **Reconfigurable Intelligent Surface (RIS)** — passive beamforming to improve secrecy performance
- **Deep Reinforcement Learning (DRL)** — joint optimization of UAV trajectory, beamforming, and antenna configuration

### Implemented Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Twin-TD3** | Twin Twin-Delayed Deep Deterministic Policy Gradient — core algorithm for joint optimization |
| **PPO** | Proximal Policy Optimization |
| **SAC** | Soft Actor-Critic |
| **TD3** | Twin Delayed DDPG |
| **DDPG** | Deep Deterministic Policy Gradient |

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

## System Model

The system considers a UAV base station equipped with FAS and RIS to serve legitimate users while maintaining communication security against eavesdroppers:

- **UAV** flies at a fixed altitude, jointly optimizing trajectory and active beamforming
- **FAS** enables flexible antenna port selection for enhanced spatial diversity
- **RIS** provides passive beamforming to boost legitimate signals and suppress eavesdropping
- **DRL agents** make real-time decisions per time slot for trajectory and beamforming optimization

## Acknowledgement

This project builds upon the [Twin-TD3](https://github.com/yjwong1999/Twin-TD3) framework for RIS-aided UAV communication. RIS simulation is based on the [SimRIS Channel Simulator](https://github.com/Brook1711/RIS_components).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
