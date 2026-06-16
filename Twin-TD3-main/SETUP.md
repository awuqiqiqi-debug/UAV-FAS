# Twin-TD3 环境配置指南

## 当前状态
**所有依赖库已安装完成** (2026-06-16)

## 快速安装

### Windows 用户
1. 双击运行 `install_deps.bat`
2. 等待安装完成

### 手动安装
```bash
cd Twin-TD3-main
pip install -r requirements.txt
```

## 验证安装

运行依赖检查脚本：
```bash
python check_deps.py
```

## 已安装依赖版本

| 包名 | 版本 |
|------|------|
| PyTorch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| NumPy | 2.1.3 |
| SciPy | 1.15.3 |
| Pandas | 2.2.3 |
| Scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| Plotly | 5.24.1 |
| OpenPyXL | 3.1.5 |
| python-pptx | 1.0.2 |
| joblib | 1.4.2 |
| pyparsing | 3.2.0 |

## 主要依赖库说明

### 核心计算
- **NumPy**: 数值计算基础库
- **SciPy**: 科学计算工具（信号处理、优化等）
- **Pandas**: 数据处理和分析

### 机器学习
- **PyTorch**: 深度学习框架（TD3/DDPG/SAC算法）
- **torchvision**: PyTorch视觉工具
- **Scikit-learn**: 机器学习工具（数据预处理、评估指标）

### 可视化
- **Matplotlib**: 基础绘图库
- **Seaborn**: 统计可视化
- **Plotly**: 交互式图表（HTML报告生成）

### 数据处理
- **OpenPyXL**: Excel文件读写（初始化位置数据）
- **python-pptx**: PowerPoint生成（教学课件）

## 常见问题

### PyTorch 安装问题
如果遇到 CUDA 版本问题，可以指定 CUDA 版本：
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 版本 (当前安装)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 中文字体问题
如果 matplotlib 显示中文乱码，需要安装中文字体：
```bash
# Windows 系统通常已有 SimHei 字体
# 如果没有，可以从网上下载 SimHei.ttf 放到 fonts 目录
```

### 内存不足
训练时如果遇到内存不足，可以：
1. 减小 batch_size
2. 减小网络层维度
3. 使用更小的 replay buffer
