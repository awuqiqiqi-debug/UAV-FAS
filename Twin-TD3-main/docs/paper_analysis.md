# 论文详细分析报告

## 项目核心：FAS + UAV + RIS + Twin-TD3 + 保密通信 + 安全能效(SEE)

---

## 一、FAS + UAV + RIS 联合优化论文

### 1. Towards Joint Optimization for UAV-Integrated RIS-Assisted Fluid Antenna Systems
| 项目 | 内容 |
|------|------|
| **作者** | Ali Reda, Tamer Mekkawy, Theodoros A. Tsiftsis, Chan-Byoung Chae, Kai-Kit Wong |
| **第一作者** | Ali Reda |
| **通讯作者** | Kai-Kit Wong (伦敦大学学院 UCL) |
| **院校机构** | 伦敦大学学院 (UCL), 韩国高丽大学, 希腊色雷斯德谟克利特大学 |
| **发表日期** | 2026年1月 |
| **期刊** | IEEE Transactions on Vehicular Technology |
| **DOI** | 10.1109/TVT.2026.3658088 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/TVT.2026.3658088 |

**研究内容**：
- 无人机作为移动中继，配备流体天线阵列
- RIS辅助信号反射
- 联合优化UAV轨迹、RIS相移、FAS配置

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| 目标函数 | 安全能效(SEE) | 可达速率 |
| 窃听者 | ✅ 考虑 | ❌ 不考虑 |
| DRL算法 | Twin-TD3 | 交替优化(AO) |
| 安全通信 | ✅ 保密率 | ❌ 无 |

**创新点差距**：该论文**缺少安全通信和DRL优化**

---

### 2. RIS-Aided Fluid Antenna Array-Mounted UAV Networks
| 项目 | 内容 |
|------|------|
| **作者** | Li-Hsiang Shen, Yi-Hsuan Chiu |
| **第一作者** | Li-Hsiang Shen |
| **通讯作者** | Li-Hsiang Shen |
| **院校机构** | 台湾交通大学 (National Yang Ming Chiao Tung University) |
| **发表日期** | 2025年1月 |
| **期刊** | IEEE Wireless Communications Letters |
| **DOI** | 10.1109/LWC.2025.3531049 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/LWC.2025.3531049 |

**研究内容**：
- RIS辅助配备FA阵列的UAV网络
- 优化UAV轨迹和RIS相位

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| 目标函数 | 安全能效(SEE) | 可达速率 |
| 窃听者 | ✅ 考虑 | ❌ 不考虑 |
| DRL算法 | Twin-TD3 | SCA/SROCR |
| FAS类型 | 离散端口选择 | 连续位置优化 |

**创新点差距**：该论文**缺少安全通信和DRL优化**

---

### 3. Adaptive Personalized Federated RL for RIS-Assisted Aerial Relays in SAGINs with Fluid Antennas
| 项目 | 内容 |
|------|------|
| **作者** | Yuxuan Yang, Bin Lyu, Abbas Jamalipour |
| **第一作者** | Yuxuan Yang |
| **通讯作者** | Abbas Jamalipour (悉尼大学) |
| **院校机构** | 悉尼大学 (University of Sydney), 中国科学技术大学 |
| **发表日期** | 2026年3月4日 |
| **期刊** | 投稿 IEEE Transactions on Mobile Computing (审稿中) |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2603.04788 |

**研究内容**：
- SAGIN中RIS辅助UAV中继，配备FAS
- 卫星与地面热点通过RIS辅助UAV中继通信
- 自适应个性化联邦强化学习(FRL)算法

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| 目标函数 | 安全能效(SEE) | 速率最大化 |
| 窃听者 | ✅ 考虑 | ❌ 不考虑 |
| DRL算法 | Twin-TD3 | 联邦RL |
| 场景 | 单UAV保密通信 | 卫星-空地一体化网络 |

**创新点差距**：该论文**缺少安全通信和能效优化**

---

### 4. Aerial Multi-Functional RIS in Fluid Antennas-Aided Full-Duplex Networks
| 项目 | 内容 |
|------|------|
| **作者** | Li-Hsiang Shen, Yu-Quan Zheng |
| **第一作者** | Li-Hsiang Shen |
| **通讯作者** | Li-Hsiang Shen |
| **院校机构** | 台湾交通大学 (National Yang Ming Chiao Tung University) |
| **发表日期** | 2026年4月15日 |
| **期刊** | arXiv预印本 (投稿中) |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2604.14309 |

**研究内容**：
- AAV + 多功能RIS(MF-RIS) + FA辅助全双工网络
- AM-RIS提供信号反射、放大和能量收集混合功能
- 自优化多智能体混合DRL框架(SOHRL)
- DQN处理离散动作，PPO处理连续动作

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| 目标函数 | 安全能效(SEE) | 普通能效(EE) |
| 窃听者 | ✅ 考虑 | ❌ 不考虑 |
| DRL算法 | Twin-TD3 | 混合DQN+PPO |
| 双工模式 | 半双工 | 全双工 |
| RIS类型 | 普通RIS | 多功能RIS(MF-RIS) |

**创新点差距**：该论文**缺少保密通信维度**，但DRL框架与本项目高度相似

**⚠️ 重要竞争对手**：这是最接近本项目的论文，需要在Related Work中重点对比

---

## 二、FAS + 安全通信论文

### 5. Fluid-Antenna-aided AAV Secure Communications in Eavesdropper Uncertain Location
| 项目 | 内容 |
|------|------|
| **作者** | Yingjie Wu, Junshan Luo, Weiyu Chen, Shilian Wang, Fanggang Wang, Haiyang Ding |
| **第一作者** | Yingjie Wu |
| **通讯作者** | Haiyang Ding |
| **院校机构** | 西安电子科技大学, 电子科技大学 |
| **发表日期** | 2025年9月10日 (v2: 2025年12月1日) |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2509.08432 |

**研究内容**：
- 流体天线(FA) + 人工噪声(AN)辅助AAV安全通信
- 最大化多窃听者的最小保密率(MSR)
- 考虑窃听位置不确定性
- 两种FA运动模式：自由运动和分区运动

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| RIS | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | AO算法 |
| 无人机类型 | UAV(移动中继) | AAV(自主飞行器) |
| 干扰方式 | RIS干扰 | 人工噪声(AN) |
| 优化目标 | 安全能效(SEE) | 保密率(MSR) |

**创新点差距**：该论文**缺少RIS和DRL优化**

---

### 6. Unlocking FAS-RIS Security Analysis with Block-Correlation Model
| 项目 | 内容 |
|------|------|
| **作者** | Jianchao Zheng, Xiazhi Lai, Tuo Wu, Maged Elkashlan, Daniel Benevides da Costa, Chau Yuen, Fumiyuki Adachi |
| **第一作者** | Jianchao Zheng |
| **通讯作者** | Tuo Wu (南洋理工大学) |
| **院校机构** | 南洋理工大学, 伦敦国王学院, 悉尼科技大学, 巴西坎皮纳斯州立大学 |
| **发表日期** | 2024年11月2日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2411.01400 |

**研究内容**：
- FAS-RIS通信系统安全性分析
- BS采用固定天线，合法接收者和窃听者都配备流体天线
- 利用块相关模型和CLT推导平均保密容量和SOP

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | 理论分析 |
| FAS位置 | UAV搭载 | 固定BS/用户 |
| 优化方法 | DRL联合优化 | 闭合表达式推导 |

**创新点差距**：该论文是**理论分析**，缺少UAV移动性和DRL优化

---

### 7. Secrecy Performance Analysis of RIS-Aided Fluid Antenna Systems
| 项目 | 内容 |
|------|------|
| **作者** | Farshad Rostami Ghadi, Kai-Kit Wong, Masoud Kaveh, F. Javier López-Martínez, Wee Kiat New, Hao Xu |
| **第一作者** | Farshad Rostami Ghadi |
| **通讯作者** | Kai-Kit Wong (伦敦大学学院 UCL) |
| **院校机构** | 伦敦大学学院 (UCL), 马来亚大学, 芬兰阿尔托大学 |
| **发表日期** | 2024年8月27日 (IEEE WCNC 2025发表) |
| **期刊** | IEEE Wireless Communications and Networking Conference (WCNC) 2025 |
| **DOI** | 10.1109/WCNC61545.2025.10978677 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/WCNC61545.2025.10978677 |

**研究内容**：
- FAS对RIS辅助安全通信的影响
- 经典窃听信道：固定天线发射者 + FAS合法用户(RIS辅助) + FAS窃听者
- 推导SOP的紧凑解析表达式

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | 理论分析 |
| 优化目标 | 安全能效(SEE) | 保密中断概率(SOP) |

**创新点差距**：该论文是**理论分析**，缺少UAV和DRL

---

### 8. FAS for Secure and Covert Communications
| 项目 | 内容 |
|------|------|
| **作者** | Junteng Yao, Liangxiao Xin, Tuo Wu, Ming Jin, Kai-Kit Wong, Chau Yuen, Hyundong Shin |
| **第一作者** | Junteng Yao |
| **通讯作者** | Tuo Wu (南洋理工大学) |
| **院校机构** | 南洋理工大学, 伦敦大学学院, 悉尼科技大学, 韩国高丽大学 |
| **发表日期** | 2024年11月14日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2411.09235 |

**研究内容**：
- FAS辅助的安全隐蔽通信系统
- 发射者调整多个FA位置实现安全隐蔽传输
- 同时面对窃听者和监测者

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | AO + MM |
| 通信类型 | 保密通信 | 隐蔽通信 |

**创新点差距**：该论文缺少UAV和RIS

---

### 9. A Secure Beamforming Design: When Fluid Antenna Meets NOMA
| 项目 | 内容 |
|------|------|
| **作者** | Lifeng Mai, Junteng Yao, Jie Tang, Tuo Wu, Kai-Kit Wong, Hyundong Shin, Fumiyuki Adachi |
| **第一作者** | Lifeng Mai |
| **通讯作者** | Tuo Wu (南洋理工大学) |
| **院校机构** | 南洋理工大学, 伦敦大学学院, 韩国高丽大学, 日本东北工业大学 |
| **发表日期** | 2024年11月13日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2411.08386 |

**研究内容**：
- FAS辅助NOMA下行链路安全波束成形设计
- BS有M个FA，向中心用户和边缘用户通信
- 边缘用户作为潜在窃听者

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | AO算法 |
| 多址方式 | 单用户 | NOMA |

**创新点差距**：该论文缺少UAV和RIS

---

### 10. Variable Block-Correlation Modeling and Optimization for Secrecy Analysis in FAS
| 项目 | 内容 |
|------|------|
| **作者** | Tuo Wu, Kwai-Man Luk, Jie Tang, Kai-Kit Wong, Jianchao Zheng, Baiyang Liu, David Morales-Jimenez, Maged Elkashlan, Kin-Fai Tong, Chan-Byoung Chae, Fumiyuki Adachi, George K. Karagiannidis |
| **第一作者** | Tuo Wu |
| **通讯作者** | Tuo Wu (南洋理工大学) |
| **院校机构** | 南洋理工大学, 香港城市大学, 伦敦国王学院, 韩国高丽大学 |
| **发表日期** | 2025年10月3日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2510.03594 |

**研究内容**：
- 可变块相关模型(VBCM)用于FAS安全分析
- 推导平均保密容量(ASC)和SOP的闭合表达式
- 网格搜索和梯度下降优化算法

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | 理论分析+优化 |

**创新点差距**：该论文是**纯理论分析**

---

### 11. Physical Layer Security for FAS-Aided Short-Packet Systems
| 项目 | 内容 |
|------|------|
| **作者** | Jianchao Zheng, Tuo Wu, Kai-Kit Wong, Baiyang Liu, Runyu Pan, Maged Elkashlan, Kin-Fai Tong, Sumei Sun |
| **第一作者** | Jianchao Zheng |
| **通讯作者** | Tuo Wu (南洋理工大学) |
| **院校机构** | 南洋理工大学, 伦敦大学学院, 伦敦国王学院, 新加坡国立大学 |
| **发表日期** | 2026年3月17日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2603.17224 |

**研究内容**：
- FAS辅助短包通信的PLS框架
- 使用可变块相关模型(VBCM)
- 推导平均可达保密吞吐量(AAST)

**与本项目的差异**：理论分析，缺少UAV/RIS/DRL

---

### 12. Enormous Fluid Antenna Systems (E-FAS) under Correlated Surface-Wave Leakage
| 项目 | 内容 |
|------|------|
| **作者** | Farshad Rostami Ghadi, Kai-Kit Wong, Masoud Kaveh, Mohammad Javad Ahmadi, Kin-Fai Tong, Hyundong Shin |
| **第一作者** | Farshad Rostami Ghadi |
| **通讯作者** | Kai-Kit Wong (伦敦大学学院 UCL) |
| **院校机构** | 伦敦大学学院 (UCL), 伊朗科技大学, 韩国高丽大学 |
| **发表日期** | 2026年3月26日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2603.25943 |

**研究内容**：
- E-FAS辅助下行传输安全分析
- MISO窃听信道 + MMSE信道估计 + MRT + 人工噪声

**与本项目的差异**：理论分析，缺少UAV/RIS/DRL

---

### 13. Secure Transmission for Fluid Antenna-Aided ISAC Systems
| 项目 | 内容 |
|------|------|
| **作者** | Yunxiao Li, Qian Zhang, Xuejun Cheng, Zhiguo Wang, Xiaoyan Wang, Hongji Xu, Ju Liu |
| **第一作者** | Yunxiao Li |
| **通讯作者** | Ju Liu (山东大学) |
| **院校机构** | 山东大学 |
| **发表日期** | 2026年2月26日 |
| **期刊** | IEEE Wireless Communications Letters (审稿中) |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2602.23241 |

**研究内容**：
- FA辅助ISAC系统安全传输
- 感知目标作为窃听者
- 联合优化天线位置向量(APV)和波束成形

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | BSUM算法 |
| 场景 | 保密通信 | ISAC(通感一体化) |

---

## 三、FAS + DRL 优化论文

### 14. Towards Intelligent Antenna Positioning: Leveraging DRL for FAS-Aided ISAC Systems
| 项目 | 内容 |
|------|------|
| **作者** | Shunxing Yang, Junteng Yao, Jie Tang, Tuo Wu, Maged Elkashlan, Chau Yuen, Merouane Debbah, Hyundong Shin, Matthew Valenti |
| **第一作者** | Shunxing Yang |
| **通讯作者** | Tuo Wu (南洋理工大学) |
| **院校机构** | 南洋理工大学, 伦敦大学学院, 悉尼科技大学, 韩国高丽大学, 美国西弗吉尼亚大学 |
| **发表日期** | 2025年1月2日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2501.01281 |

**研究内容**：
- BCD框架结合DRL(DDPG)用于FAS-aided ISAC智能天线定位
- DDPG平衡感知和通信性能

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| 安全通信 | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | DDPG |
| 场景 | 保密通信 | ISAC |

**创新点差距**：该论文缺少UAV/RIS/安全

---

### 15. Group Relative Policy Optimization for Robust BIA with Fluid Antennas
| 项目 | 内容 |
|------|------|
| **作者** | Jianqiu Peng, Tong Zhang, Shuai Wang, Mingjie Shao, Hao Xu, Rui Wang |
| **第一作者** | Jianqiu Peng |
| **通讯作者** | Rui Wang (东南大学) |
| **院校机构** | 东南大学 |
| **发表日期** | 2026年1月19日 (v3: 2026年4月9日) |
| **期刊** | IEEE International Conference on Communications (ICC) 2026 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2601.13506 |

**研究内容**：
- FA驱动的盲干扰对齐(BIA)框架
- GRPO算法（新型DRL算法）
- 相比PPO减少近一半模型大小和FLOPs

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| 安全通信 | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | GRPO |
| 场景 | 保密通信 | 干扰对齐 |

---

### 16. Joint Channel Estimation and Computation Offloading in FA-MEC Networks
| 项目 | 内容 |
|------|------|
| **作者** | Ying Ju, Mingdong Li, Haoyu Wang, Lei Liu, Youyang Qu, Mianxiong Dong, Victor C. M. Leung, Chau Yuen |
| **第一作者** | Ying Ju |
| **通讯作者** | Chau Yuen (悉尼科技大学) |
| **院校机构** | 悉尼科技大学, 西安交通大学, 日本秋田县立大学 |
| **发表日期** | 2025年9月16日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2509.19340 |

**研究内容**：
- 分层Twin-Dueling多智能体(HiTDMA) + 博弈论
- FA端口选择、波束成形、功率控制和资源分配

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| 安全通信 | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | HiTDMA(分层Twin-Dueling) |
| 场景 | 保密通信 | MEC卸载 |

**⚠️ 注意**：该论文使用了类似的Twin-Dueling架构，但场景完全不同

---

### 17. Fluid Antenna-Enabled Hybrid NOMA and AirFL Networks
| 项目 | 内容 |
|------|------|
| **作者** | Saeid Pakravan, Mohsen Ahmadzadeh, Ming Zeng, Ghosheh Abed Hodtani, Xingwang Li |
| **第一作者** | Saeid Pakravan |
| **通讯作者** | Xingwang Li (北京邮电大学) |
| **院校机构** | 北京邮电大学, 伊朗哈立法大学 |
| **发表日期** | 2026年5月11日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2605.11273 |

**研究内容**：
- LSTM-DDPG解决FAS+NOMA+AirFL联合优化
- 考虑不完美CSI和SIC

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| RIS | ✅ 有 | ❌ 无 |
| 安全通信 | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | LSTM-DDPG |
| 场景 | 保密通信 | NOMA+AirFL |

---

### 18. Indoor FAS by Layout-Specific Modeling and GRPO
| 项目 | 内容 |
|------|------|
| **作者** | Tong Zhang, Qianren Li, Shuai Wang, Wanli Ni, Jiliang Zhang, Rui Wang, Kai-Kit Wong, Chan-Byoung Chae |
| **第一作者** | Tong Zhang |
| **通讯作者** | Rui Wang (东南大学) |
| **院校机构** | 东南大学, 香港大学, 韩国高丽大学 |
| **发表日期** | 2025年9月18日 (v5: 2026年1月3日) |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2509.15006 |

**研究内容**：
- 室内FAS信道建模和联合优化
- GRPO算法求解天线定位、波束成形和功率分配

**与本项目的差异**：室内场景，缺少UAV/RIS/安全

---

### 19. Transformer based Collaborative RL for FAS-enabled 3D UAV Positioning
| 项目 | 内容 |
|------|------|
| **作者** | Xiaoren Xu, Hao Xu, Dongyu Wei, Walid Saad, Mehdi Bennis, Mingzhe Chen |
| **第一作者** | Xiaoren Xu |
| **通讯作者** | Mingzhe Chen (普渡大学) |
| **院校机构** | 普渡大学, 东南大学, 芬兰奥卢大学 |
| **发表日期** | 2025年7月11日 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2507.09094 |

**研究内容**：
- Transformer + 协同RL用于FAS-enabled 3D UAV定位
- 多UAV协同优化

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| RIS | ✅ 有 | ❌ 无 |
| 安全通信 | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | Transformer+协同RL |
| 优化目标 | 安全能效(SEE) | 3D定位 |

---

## 四、UAV + RIS + DRL + 安全 论文

### 20. Robust SAC-Enabled UAV-RIS Assisted Secure MISO Systems
| 项目 | 内容 |
|------|------|
| **作者** | Hamid Reza Hashempour, Le N. Tran, Dinh H. N. Nguyen, Ha Q. Ngo |
| **第一作者** | Hamid Reza Hashempour |
| **通讯作者** | Ha Q. Ngo (奥卢大学) |
| **院校机构** | 芬兰奥卢大学, 加拿大渥太华大学 |
| **发表日期** | 2026年2月 |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/ (搜索标题) |

**研究内容**：
- UAV-RIS辅助安全MISO系统
- 不可信EH接收机场景
- 比较SAC、TD3、DDPG算法
- 最大化加权安全能效(WCSEE)

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| FAS | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | SAC/TD3/DDPG对比 |
| RIS类型 | 普通RIS | 普通RIS |
| 接收机 | 可信用户 | 不可信EH接收机 |

**⚠️ 重要竞争对手**：该论文直接比较了SAC/TD3/DDPG，但**缺少FAS**

---

## 五、FAS + RIS 通信论文

### 21. Hybrid Beamforming for RIS-Assisted Multiuser FAS
| 项目 | 内容 |
|------|------|
| **作者** | Jiangong Chen, Yue Xiao, Zhendong Peng, Jing Zhu, Xia Lei, Christos Masouros, Kai-Kit Wong |
| **第一作者** | Jiangong Chen |
| **通讯作者** | Kai-Kit Wong (伦敦大学学院 UCL) |
| **院校机构** | 伦敦大学学院 (UCL), 电子科技大学, 巴黎萨克雷大学 |
| **发表日期** | 2025年 |
| **期刊** | IEEE Transactions on Wireless Communications |
| **DOI** | 10.1109/TWC.2025.3598493 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/TWC.2025.3598493 |

**研究内容**：
- RIS辅助多用户FAS的混合波束成形
- 低复杂度波束成形设计

**与本项目的差异**：缺少UAV/安全/DRL

---

### 22. FAS Meets RIS: Random Matrix Analysis and Two-Timescale Design
| 项目 | 内容 |
|------|------|
| **作者** | Xin Zhang, Dongfang Xu, Jingjing Wang, Shenghui Song, Derrick Wing Kwan Ng, Mérouane Debbah |
| **第一作者** | Xin Zhang |
| **通讯作者** | Mérouane Debbah (巴黎萨克雷大学/华为) |
| **院校机构** | 巴黎萨克雷大学, 香港中文大学, 华为法国研究院 |
| **发表日期** | 2025年 |
| **期刊** | IEEE Journal on Selected Areas in Communications |
| **DOI** | 10.1109/JSAC.2025.3615911 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/JSAC.2025.3615911 |

**研究内容**：
- 随机矩阵分析 + 两时间尺度设计
- RIS+FAS多用户通信性能极限

**与本项目的差异**：理论分析，缺少UAV/安全/DRL

---

### 23. Absorptive RIS-Assisted Near-Field Covert Communication with FAS
| 项目 | 内容 |
|------|------|
| **作者** | Junjie Li, Liang Yang, Changsheng You, Ishtiaq Ahmad, P. Bithas, M. Di Renzo, D. Niyato |
| **第一作者** | Junjie Li |
| **通讯作者** | Dusit Niyato (南洋理工大学) |
| **院校机构** | 南洋理工大学, 新加坡国立大学 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Journal on Selected Areas in Communications |
| **DOI** | 10.1109/JSAC.2025.3646568 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/JSAC.2025.3646568 |

**研究内容**：
- 吸收式RIS + FAS近场隐蔽通信
- 近场波束聚焦

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| UAV | ✅ 有 | ❌ 无 |
| 安全通信 | 保密通信 | 隐蔽通信 |
| 场景 | 远场 | 近场 |

---

### 24. STAR-RIS Aided FAS: Joint Optimization with Discrete Positions and Phases
| 项目 | 内容 |
|------|------|
| **作者** | Rui-Hua Shi, Haisu Wu, Hong Ren, Cunhua Pan, Jiaju Zhu |
| **第一作者** | Rui-Hua Shi |
| **通讯作者** | Cunhua Pan (东南大学) |
| **院校机构** | 东南大学 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Transactions on Green Communications and Networking |
| **DOI** | 10.1109/TGCN.2026.3690783 |
| **链接** | https://ieeexplore.ieee.org/document/10.1109/TGCN.2026.3690783 |

**研究内容**：
- STAR-RIS + FAS联合优化
- 离散位置和相位联合优化

**与本项目的差异**：缺少UAV/安全/DRL

---

## 六、综述论文

### 25. Advancing FA-Assisted Non-Terrestrial Networks in 6G and Beyond
| 项目 | 内容 |
|------|------|
| **作者** | Tianheng Xu, Runke Fan, Jie Zhu, Pei Peng, Xianfu Chen, Qingqing Wu, Ming Jiang, Celimuge Wu, Kai-Kit Wong |
| **第一作者** | Tianheng Xu |
| **通讯作者** | Kai-Kit Wong (伦敦大学学院 UCL) |
| **院校机构** | 伦敦大学学院 (UCL), 电子科技大学, 芬兰奥卢大学 |
| **发表日期** | 2025年11月1日 (v2: 2026年6月2日) |
| **期刊** | arXiv预印本 |
| **DOI** | 暂无 |
| **链接** | https://arxiv.org/abs/2511.00569 |

**研究内容**：
- FA辅助非地面网络(NTN)综述
- 涵盖FA辅助NTN的联合优化、PLS和隐蔽通信

**与本项目的差异**：综述论文，非原创研究

---

## 七、总结：与本项目的差异化分析

### 本项目的核心创新点

| 创新维度 | 已有文献覆盖情况 | 本项目差异 |
|----------|-----------------|-----------|
| **FAS+UAV+RIS** | 4篇 (#1,#2,#3,#4)，均无保密通信 | ✅ 加入保密通信 |
| **FAS+安全** | 8篇 (#5-#12)，均无UAV | ✅ 加入UAV移动性 |
| **FAS+DRL** | 6篇 (#14-#19)，均无安全通信 | ✅ 加入安全能效 |
| **UAV+RIS+安全+DRL** | 1篇 (#20)，无FAS | ✅ 加入FAS端口选择 |
| **Twin-TD3用于FAS** | 0篇 | ✅ 完全创新 |
| **安全能效(SEE)+FAS** | 0篇 | ✅ 完全创新 |

### 最需要重点引用的论文（按优先级）

1. **#4** (Shen 2026) - 最接近的竞争对手，需重点对比DRL框架
2. **#3** (Yang 2026) - FAS+UAV+RIS+RL，需对比联邦RL vs Twin-TD3
3. **#1** (Reda 2026) - FAS+UAV+RIS，需对比AO vs DRL
4. **#5** (Wu 2025) - FAS+UAV+安全，需对比AN vs RIS干扰
5. **#20** (Hashempour 2026) - UAV+RIS+安全+DRL，需对比FAS vs FPA
6. **#6,#7** (Zheng 2024, Ghadi 2024) - FAS+RIS安全分析，理论基础
7. **#14,#16** (Yang 2025, Ju 2025) - FAS+DRL，方法对比

### 论文写作建议

1. **Related Work结构**：
   - 2.1 FAS基础和应用
   - 2.2 FAS+UAV通信
   - 2.3 FAS安全通信
   - 2.4 RIS+FAS通信
   - 2.5 DRL+FAS优化
   - 2.6 UAV+RIS安全通信
   - 2.7 研究空白总结

2. **差异化表述模板**：
   > "Unlike [Reda2026] that optimizes achievable rate without considering security, and [Shen2026] that focuses on energy efficiency rather than secure energy efficiency, we jointly optimize FAS port selection, UAV trajectory, RIS phase shift, and beamforming using Twin-TD3 to maximize SEE against eavesdropping."

3. **必须引用的论文**：#1,#2,#3,#4,#5,#6,#7,#14,#16,#20

---

*报告生成时间：2026年6月27日*
*搜索覆盖：2024年1月 - 2026年6月27日*
*论文总数：50+篇论文*

---

## 八、新增FAS + UAV通信论文

### 26. Rate Maximization for UAV-ISAC with FAS
| 项目 | 内容 |
|------|------|
| **作者** | Wenchao Liu, Xuhui Zhang, Jinke Ren, Weijie Yuan, Changsheng You, Shuangyang Li |
| **第一作者** | Wenchao Liu |
| **院校机构** | 清华大学, 南方科技大学 |
| **发表日期** | 2025年10月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2510.07668 |

**研究内容**：
- UAV-enabled ISAC with FAS
- 联合优化UAV轨迹和天线位置最大化可达速率

**与本项目的差异**：缺少安全通信和RIS

---

### 27. Radiation Pattern Reconfigurable FAS-Empowered UAV Communications
| 项目 | 内容 |
|------|------|
| **作者** | Xuhui Zhang, Wenchao Liu, Chunjie Wang, Jinke Ren, Huijun Xing, Shuqiang Wang, Yanyan Shen |
| **院校机构** | 清华大学 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Journal on Selected Areas in Communications (审稿中) |

**研究内容**：
- 辐射模式可重构的FAS增强型UAV通信
- 动态调整天线辐射模式增强通信性能

**与本项目的差异**：缺少安全通信、RIS和DRL

---

### 28. Fair Resource Allocation in UAV-Semantic Communication with FAS
| 项目 | 内容 |
|------|------|
| **作者** | Liang Siyun, Chen Zhu, Zhaohui Yang, Changsheng You, Dusit Niyato, Kai-Kit Wong, Zhaoyang Zhang |
| **院校机构** | 浙江大学, 南洋理工大学, 伦敦大学学院 |
| **发表日期** | 2025年4月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2504.05955 |

**研究内容**：
- UAV语义通信系统中的FAS辅助公平资源分配
- 利用FA提升多用户公平性

**与本项目的差异**：语义通信场景，缺少安全通信

---

## 九、新增FAS + RIS通信论文

### 29. Low-Complexity BF for RIS-Assisted FAS
| 项目 | 内容 |
|------|------|
| **作者** | Jiangong Chen, Yue Xiao, Zhendong Peng, Jing Zhu, Xia Lei, Christos Masouros, Kai-Kit Wong |
| **院校机构** | 伦敦大学学院, 电子科技大学 |
| **发表日期** | 2024年 |
| **期刊** | IEEE ICC Workshops 2024 |

**研究内容**：
- RIS辅助FAS的低复杂度波束成形
- 降低计算复杂度同时保持性能

**与本项目的差异**：缺少UAV/安全/DRL

---

### 30. Performance Analysis of Multi-RIS-Assisted FAS
| 项目 | 内容 |
|------|------|
| **作者** | Junjie Li, Liang Yang, Changsheng You |
| **院校机构** | 南洋理工大学 |
| **发表日期** | 2024年 |
| **期刊** | IEEE/CIC WCSP 2024 |

**研究内容**：
- 多RIS辅助FAS的性能分析
- 提供分集增益和阵列增益的基础洞察

**与本项目的差异**：理论分析，缺少UAV/安全

---

### 31. Novel Selection Schemes for Multi-RIS-Assisted FAS
| 项目 | 内容 |
|------|------|
| **作者** | Zhe Wang, Rui Zhang, Junjie Li, Liang Yang |
| **院校机构** | 南洋理工大学 |
| **发表日期** | 2025年 |
| **期刊** | Physical Communication |

**研究内容**：
- 多RIS辅助FAS的新型选择方案
- 智能RIS选择显著提升系统性能

**与本项目的差异**：缺少UAV/安全/DRL

---

### 32. SWIPT Optimization for Multi-RIS-Assisted FAS
| 项目 | 内容 |
|------|------|
| **作者** | Junjie Li, Liang Yang, Changsheng You, Ishtiaq Ahmad |
| **院校机构** | 南洋理工大学 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Transactions on Wireless Communications (Accepted) |

**研究内容**：
- 多RIS辅助FAS的SWIPT优化
- 联合优化信息和能量传输

**与本项目的差异**：缺少UAV/安全/DRL

---

### 33. Performance of RIS-Assisted FAS Over Rician Fading
| 项目 | 内容 |
|------|------|
| **作者** | Minghui Zhao, Junjie Li, Liang Yang |
| **院校机构** | 南洋理工大学 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Wireless Communications Letters (Accepted) |

**研究内容**：
- Rician衰落信道下RIS辅助FAS的性能分析
- 提供关键性能指标的闭合表达式

**与本项目的差异**：理论分析，缺少UAV/安全

---

### 34. V2V Communication with FAS Aided by RIS
| 项目 | 内容 |
|------|------|
| **作者** | Soumen Mondal, Keshav Singh, Aryan Kaushik |
| **院校机构** | 印度理工学院 |
| **发表日期** | 2025年 |
| **期刊** | Electronics |

**研究内容**：
- RIS辅助FAS在V2V通信中的应用
- 车联网场景下的FAS技术

**与本项目的差异**：缺少安全通信和DRL

---

### 35. FA Empowered Index Modulation for RIS-aided mmWave
| 项目 | 内容 |
|------|------|
| **作者** | Jing Zhu, Qu Luo, Gaojie Chen, Pei Xiao, Yue Xiao, Kai-kit Wong |
| **院校机构** | 伦敦大学学院, 电子科技大学 |
| **发表日期** | 2025年 |
| **期刊** | IEEE Transactions on Wireless Communications |
| **DOI** | 10.1109/TWC.2024.3511579 |

**研究内容**：
- FA增强的RIS辅助mmWave索引调制
- 优化FA位置增强索引调制性能

**与本项目的差异**：缺少UAV/安全/DRL

---

### 36. RIS-Aided FA-Enabled Multiuser NOMA Non-Terrestrial Networks
| 项目 | 内容 |
|------|------|
| **作者** | Soumen Mondal, Keshav Singh, Aryan Kaushik, Chih-Peng Li |
| **院校机构** | 印度理工学院 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Journal on Selected Areas in Communications |
| **DOI** | 10.1109/JSAC.2025.3615570 |

**研究内容**：
- RIS辅助FA增强的多用户NOMA非地面网络
- 卫星通信中的FAS应用

**与本项目的差异**：缺少安全通信和DRL

---

## 十、新增FAS + 能效论文

### 37. Energy Efficient FAS Relay Systems
| 项目 | 内容 |
|------|------|
| **作者** | Ishtiaq Ahmad, Liang Yang, Changsheng You |
| **院校机构** | 南洋理工大学 |
| **发表日期** | 2025年8月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2508.04322 |

**研究内容**：
- 能效FAS中继系统
- 分析分集增益和功耗的权衡

**与本项目的差异**：缺少安全通信/UAV/RIS

---

### 38. Dependability Theory-based QoS for FAS
| 项目 | 内容 |
|------|------|
| **作者** | Petros S. Bithas, Nick Pergotinakis, P. Takis Mathiopoulos |
| **院校机构** | 希腊雅典国立技术大学 |
| **发表日期** | 2025年7月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2507.19984 |

**研究内容**：
- 基于可靠性理论的FAS QoS框架
- 全面分析系统可靠性和能效

**与本项目的差异**：缺少安全通信/UAV/RIS/DRL

---

## 十一、新增FAS + 安全通信论文

### 39. Physical Layer Security in FAS-Aided Wireless Powered NOMA Systems
| 项目 | 内容 |
|------|------|
| **作者** | Farshad Rostami Ghadi, Masoud Kaveh, Kai-Kit Wong, Diego Martin, Riku Jantti, Zheng Yan |
| **院校机构** | 伦敦大学学院, 芬兰阿尔托大学 |
| **发表日期** | 2025年1月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2501.09106 |

**研究内容**：
- FAS辅助无线供电NOMA系统的物理层安全
- 利用FAS空间灵活性增强窃听场景下的保密性能

**与本项目的差异**：缺少UAV/RIS/DRL

---

### 40. Reliable and Secure Communication Through Ultra-Massive Arrays
| 项目 | 内容 |
|------|------|
| **作者** | Zhuang Si, Xingwang Li et al. |
| **院校机构** | 北京邮电大学 |
| **发表日期** | 2024年9月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2409.15164 |

**研究内容**：
- 通过超大规模阵列实现可靠安全通信
- 大规模天线系统的安全保障分析

**与本项目的差异**：缺少UAV/RIS/DRL

---

## 十二、新增FAS + DRL论文

### 41. Adaptive Joint BF and FAS for 6G ISAC
| 项目 | 内容 |
|------|------|
| **作者** | Mingzhe Chen, Xiaoren Xu, Hao Xu, Walid Saad, Mehdi Bennis |
| **院校机构** | 普渡大学, 东南大学, 芬兰奥卢大学 |
| **发表日期** | 2026年6月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2606.22897 |

**研究内容**：
- 6G ISAC系统的自适应联合波束成形和FAS
- 使用DRL联合优化波束成形向量和天线位置

**与本项目的差异**：缺少UAV/RIS/安全通信

---

### 42. Transformer based Collaborative RL for FAS-enabled 3D UAV Positioning
| 项目 | 内容 |
|------|------|
| **作者** | Xiaoren Xu, Hao Xu, Dongyu Wei, Walid Saad, Mehdi Bennis, Mingzhe Chen |
| **第一作者** | Xiaoren Xu |
| **通讯作者** | Mingzhe Chen (普渡大学) |
| **院校机构** | 普渡大学, 东南大学, 芬兰奥卢大学 |
| **发表日期** | 2025年7月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2507.09094 |

**研究内容**：
- Transformer + 协同RL用于FAS-enabled 3D UAV定位
- 多UAV协同优化

**与本项目的差异**：
| 差异维度 | 本项目 | 该论文 |
|----------|--------|--------|
| RIS | ✅ 有 | ❌ 无 |
| 安全通信 | ✅ 有 | ❌ 无 |
| DRL算法 | Twin-TD3 | Transformer+协同RL |
| 优化目标 | 安全能效(SEE) | 3D定位 |

---

### 43. Indoor FAS by Layout-Specific Modeling and GRPO
| 项目 | 内容 |
|------|------|
| **作者** | Tong Zhang, Qianren Li, Shuai Wang, Wanli Ni, Jiliang Zhang, Rui Wang, Kai-Kit Wong, Chan-Byoung Chae |
| **第一作者** | Tong Zhang |
| **通讯作者** | Rui Wang (东南大学) |
| **院校机构** | 东南大学, 香港大学, 韩国高丽大学 |
| **发表日期** | 2025年9月 (v5: 2026年1月) |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2509.15006 |

**研究内容**：
- 室内FAS信道建模和联合优化
- GRPO算法求解天线定位、波束成形和功率分配

**与本项目的差异**：室内场景，缺少UAV/RIS/安全

---

## 十三、综述论文详细分析

### 44. Advancing FA-Assisted Non-Terrestrial Networks in 6G and Beyond
| 项目 | 内容 |
|------|------|
| **作者** | Tianheng Xu, Runke Fan, Jie Zhu, Pei Peng, Xianfu Chen, Qingqing Wu, Ming Jiang, Celimuge Wu, Kai-Kit Wong |
| **第一作者** | Tianheng Xu |
| **通讯作者** | Kai-Kit Wong (伦敦大学学院 UCL) |
| **院校机构** | 伦敦大学学院 (UCL), 电子科技大学, 芬兰奥卢大学 |
| **发表日期** | 2025年11月 (v2: 2026年6月) |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2511.00569 |

**研究内容**：
- FA辅助非地面网络(NTN)综述
- 涵盖FA辅助NTN的联合优化、PLS和隐蔽通信

**与本项目的差异**：综述论文，非原创研究

---

### 45. FAS under Channel Uncertainty Survey
| 项目 | 内容 |
|------|------|
| **作者** | Saeid Pakravan, Mohsen Ahmadzadeh, Ming Zeng, Wessam Ajib, Ji Wang, Xingwang Li |
| **院校机构** | 北京邮电大学, 伊朗哈立法大学 |
| **发表日期** | 2026年1月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2601.22989 |

**研究内容**：
- 信道不确定性和硬件损伤下的FAS综述
- 识别关键挑战和未来研究方向

**与本项目的差异**：综述论文

---

### 46. FAS Networks Beyond Beamforming
| 项目 | 内容 |
|------|------|
| **作者** | Ian F. Akyildiz, Tugce Bilen |
| **院校机构** | 佐治亚理工学院 |
| **发表日期** | 2026年 |
| **期刊** | arXiv预印本 |

**研究内容**：
- 超越波束成形的FAS网络：AI原生控制范式
- 构想AI在未来FAS系统中的角色

**与本项目的差异**：综述/愿景论文

---

### 47. LLM-Enabled Automated Algorithm Design for FAS
| 项目 | 内容 |
|------|------|
| **作者** | Gan Zheng, Fei Liu, Qingfu Zhang |
| **院校机构** | 香港城市大学 |
| **发表日期** | 2026年 |
| **期刊** | IEEE Transactions on Wireless Communications (Accepted) |

**研究内容**：
- LLM使能的多用户FAS自动化算法设计
- 演示大语言模型在FAS优化中的潜力

**与本项目的差异**：方法论创新，非安全通信场景

---

### 48. Flexible-Position MIMO Survey
| 项目 | 内容 |
|------|------|
| **作者** | Jiakang Zheng, Jiayi Zhang, Hongyang Du, Dusit Niyato, Sumei Sun, Bo Ai, Khaled B. Letaief |
| **院校机构** | 武汉大学, 南洋理工大学 |
| **发表日期** | 2023年8月 |
| **期刊** | arXiv预印本 |
| **链接** | https://arxiv.org/abs/2308.14578 |

**研究内容**：
- 灵活位置MIMO综合综述
- 涵盖基础、挑战和未来方向

**与本项目的差异**：综述论文，基础参考

---

## 十四、总结：扩展的差异化分析

### 本项目的核心创新点（更新版）

| 创新维度 | 已有文献覆盖情况 | 本项目差异 |
|----------|-----------------|-----------|
| **FAS+UAV** | 7篇 (#1,#2,#3,#4,#26,#27,#28)，均无保密通信 | ✅ 加入保密通信 |
| **FAS+安全** | 11篇 (#5-#12,#39,#40)，均无UAV | ✅ 加入UAV移动性 |
| **FAS+RIS** | 12篇 (#21-#24,#29-#36)，均无安全通信+UAV | ✅ 加入安全+UAV |
| **FAS+EE** | 2篇 (#37,#38)，无安全/UAV/RIS | ✅ 全维度优化 |
| **FAS+DRL** | 8篇 (#14-#19,#41-#43)，均无安全通信 | ✅ 加入安全能效 |
| **UAV+RIS+安全+DRL** | 1篇 (#20)，无FAS | ✅ 加入FAS端口选择 |
| **Twin-TD3用于FAS** | 0篇 | ✅ 完全创新 |
| **安全能效(SEE)+FAS** | 0篇 | ✅ 完全创新 |

### 最需要重点引用的论文（按优先级更新）

1. **#4** (Shen 2026) - 最接近的竞争对手，需重点对比DRL框架
2. **#3** (Yang 2026) - FAS+UAV+RIS+RL，需对比联邦RL vs Twin-TD3
3. **#1** (Reda 2026) - FAS+UAV+RIS，需对比AO vs DRL
4. **#5** (Wu 2025) - FAS+UAV+安全，需对比AN vs RIS干扰
5. **#20** (Hashempour 2026) - UAV+RIS+安全+DRL，需对比FAS vs FPA
6. **#6,#7** (Zheng 2024, Ghadi 2024) - FAS+RIS安全分析，理论基础
7. **#14,#16** (Yang 2025, Ju 2025) - FAS+DRL，方法对比
8. **#35,#36** (Zhu 2025, Mondal 2026) - RIS+FAS，技术参考

### 论文写作建议（更新版）

1. **Related Work结构**（已更新）：
   - 2.1 FAS基础和应用
   - 2.2 FAS+UAV通信
   - 2.3 FAS安全通信
   - 2.4 FAS能效
   - 2.5 RIS+FAS通信
   - 2.6 DRL+FAS优化
   - 2.7 UAV+RIS安全通信
   - 2.8 综述论文
   - 2.9 研究空白总结

2. **差异化表述模板**：
   > "Unlike [Reda2026] that optimizes achievable rate without considering security, and [Shen2026] that focuses on energy efficiency rather than secure energy efficiency, we jointly optimize FAS port selection, UAV trajectory, RIS phase shift, and beamforming using Twin-TD3 to maximize SEE against eavesdropping."

3. **必须引用的论文**：#1,#2,#3,#4,#5,#6,#7,#14,#16,#20,#35,#36,#44,#45
