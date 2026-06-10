"""SAC (Soft Actor-Critic) 算法实现
关键特性：最大熵目标 + 自动温度调节 + 随机策略（天然探索）
"""
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class SACActorNetwork(nn.Module):
    """SAC Actor: 输出均值和对数标准差，采样随机动作"""
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                 n_actions, name, chkpt_dir):
        super(SACActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_SAC')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        self.mu = nn.Linear(self.fc4_dims, self.n_actions)
        self.log_sig = nn.Linear(self.fc4_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)

        mu = self.mu(x)
        log_sig = self.log_sig(x)
        log_sig = T.clamp(log_sig, LOG_SIG_MIN, LOG_SIG_MAX)

        return mu, log_sig

    def sample_normal(self, state, reparameterize=True):
        """从高斯分布采样动作"""
        mu, log_sig = self.forward(state)
        sig = log_sig.exp()
        normal = Normal(mu, sig)

        if reparameterize:
            actions = normal.rsample()  # 重参数化采样
        else:
            actions = normal.sample()

        action = T.tanh(actions)

        # 计算log概率（用于熵计算）
        log_probs = normal.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + EPSILON)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        if T.cuda.is_available():
            self.load_state_dict(T.load(load_file))
        else:
            self.load_state_dict(T.load(load_file, map_location=T.device('cpu')))


class SACCriticNetwork(nn.Module):
    """SAC Critic: 双Q网络"""
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                 n_actions, name, chkpt_dir):
        super(SACCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_SAC')

        # Q1
        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.bn4 = nn.LayerNorm(self.fc4_dims)
        self.q = nn.Linear(self.fc4_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        sa = T.cat([state, action], dim=1)
        x = self.fc1(sa)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        if T.cuda.is_available():
            self.load_state_dict(T.load(load_file))
        else:
            self.load_state_dict(T.load(load_file, map_location=T.device('cpu')))


class SACAgent:
    """SAC智能体：最大熵强化学习"""
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, layer3_size=256, layer4_size=128,
                 batch_size=64, agent_name='default'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        # 自动温度调节
        self.target_entropy = -0.5 * n_actions  # 目标熵 (更宽松)
        self.log_alpha = T.zeros(1, requires_grad=True, device='cuda:0' if T.cuda.is_available() else 'cpu')
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)

        # Actor
        self.actor = SACActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                     layer3_size, layer4_size, n_actions,
                                     name='Actor_' + agent_name,
                                     chkpt_dir=env.data_manager.store_path)

        # Twin Critics
        self.critic_1 = SACCriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                         layer3_size, layer4_size, n_actions,
                                         name='Critic_1_' + agent_name,
                                         chkpt_dir=env.data_manager.store_path)
        self.critic_2 = SACCriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                         layer3_size, layer4_size, n_actions,
                                         name='Critic_2_' + agent_name,
                                         chkpt_dir=env.data_manager.store_path)

        # Target Critics
        self.target_critic_1 = SACCriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                                layer3_size, layer4_size, n_actions,
                                                name='TargetCritic_1_' + agent_name,
                                                chkpt_dir=env.data_manager.store_path)
        self.target_critic_2 = SACCriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                                layer3_size, layer4_size, n_actions,
                                                name='TargetCritic_2_' + agent_name,
                                                chkpt_dir=env.data_manager.store_path)

        # Replay Buffer
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # 初始化target networks
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def choose_action(self, observation):
        """选择动作（随机采样，无需额外噪声）"""
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        self.actor.train()
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).unsqueeze(1).to(self.actor.device)
        done = T.tensor(done, dtype=T.float).unsqueeze(1).to(self.actor.device)

        alpha = self.log_alpha.exp()

        # 更新Critic
        with T.no_grad():
            new_action, log_probs = self.actor.sample_normal(new_state, reparameterize=True)
            q1_new = self.target_critic_1(new_state, new_action)
            q2_new = self.target_critic_2(new_state, new_action)
            q_new = T.min(q1_new, q2_new) - alpha * log_probs
            target_q = reward + self.gamma * q_new * (1 - done)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()

        # 更新Actor
        new_action, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        q_new = T.min(q1_new, q2_new)

        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # 更新温度alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 更新target networks
        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self, load_file_actor='', load_file_critic_1='', load_file_critic_2=''):
        if load_file_actor:
            self.actor.load_checkpoint(load_file_actor)
        if load_file_critic_1:
            self.critic_1.load_checkpoint(load_file_critic_1)
            self.target_critic_1.load_checkpoint(load_file_critic_1)
        if load_file_critic_2:
            self.critic_2.load_checkpoint(load_file_critic_2)
            self.target_critic_2.load_checkpoint(load_file_critic_2)


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, new_states, dones
