import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.networks.actor_critic import ActorNetwork, CriticNetwork


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)


class AWGNActionNoise(object):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x


class ReplayBuffer(object):
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
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, layer3_size=256, layer4_size=128, batch_size=64,
                 update_actor_interval=2, noise='AWGN', agent_name='default', load_file='',
                 n_discrete_dims=0):
        self.load_file = load_file
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.layer4_size = layer4_size
        self.gamma = gamma
        self.tau = tau
        self.n_discrete_dims = n_discrete_dims
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                  name='Actor_' + agent_name, chkpt_dir=env.data_manager.store_path,
                                  n_discrete_dims=n_discrete_dims)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                    name='Critic_1_' + agent_name, chkpt_dir=env.data_manager.store_path)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                    name='Critic_2_' + agent_name, chkpt_dir=env.data_manager.store_path)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                         name='TargetActor_' + agent_name, chkpt_dir=env.data_manager.store_path,
                                         n_discrete_dims=n_discrete_dims)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                           name='TargetCritic_1_' + agent_name, chkpt_dir=env.data_manager.store_path)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                           name='TargetCritic_2_' + agent_name, chkpt_dir=env.data_manager.store_path)
        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_actions))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu=np.zeros(n_actions))
        # tau = 1 means copy parameters to target
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, greedy=0.5, epsilon=1):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(greedy * self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.critic_1.eval()
        self.critic_2.eval()

        target_actions = self.target_actor.forward(new_state)

        # Target Policy Smoothing: 对连续维度加噪正则化，防止Critic过估计
        if self.n_discrete_dims > 0:
            noise = T.randn_like(target_actions[:, self.n_discrete_dims:]) * 0.2
            noise = T.clamp(noise, -0.5, 0.5)
            smoothed = target_actions[:, self.n_discrete_dims:] + noise
            smoothed = T.clamp(smoothed, -1.0, 1.0)
            target_actions = T.cat([target_actions[:, :self.n_discrete_dims], smoothed], dim=-1)
        else:
            noise = T.randn_like(target_actions) * 0.2
            noise = T.clamp(noise, -0.5, 0.5)
            target_actions = T.clamp(target_actions + noise, -1.0, 1.0)

        critic_value_1_ = self.target_critic_1.forward(new_state, target_actions)
        critic_value_2_ = self.target_critic_2.forward(new_state, target_actions)

        critic_value_1 = self.critic_1.forward(state, action)
        critic_value_2 = self.critic_2.forward(state, action)

        critic_value_ = T.min(critic_value_1_, critic_value_2_)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j].detach()*done[j])
        target = T.tensor(target).to(self.critic_1.device)
        target = target.view(self.batch_size, 1)

        self.critic_1.train()
        self.critic_2.train()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        critic_1_loss = F.mse_loss(target, critic_value_1)
        critic_2_loss = F.mse_loss(target, critic_value_2)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.critic_1.eval()
        self.critic_2.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_q1_loss = self.critic_1.forward(state, mu)
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self, load_file_actor='', load_file_critic_1='', load_file_critic_2=''):
        self.actor.load_checkpoint(load_file=load_file_actor)
        self.target_actor.load_checkpoint(load_file=load_file_actor)
        self.critic_1.load_checkpoint(load_file=load_file_critic_1)
        self.critic_2.load_checkpoint(load_file=load_file_critic_2)
        self.target_critic_1.load_checkpoint(load_file=load_file_critic_1)
        self.target_critic_2.load_checkpoint(load_file=load_file_critic_2)
