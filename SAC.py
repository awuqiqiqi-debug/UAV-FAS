import torch.nn.functional as F
import numpy as np
import torch
import copy
import torch.nn as nn
from torch.distributions import Normal
import math



class SAC(object):
    def __init__(self,env,**kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.user_num = env.user_num
        self.tau = 0.005

        self.actor = Actor(self.state_dim,self.action_dim,self.hidden_size).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim,self.action_dim,self.hidden_size).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(),lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e2), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)


    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (
                        target_Q - self.alpha * log_pi_a_next)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, model_path, ele, beta, with_reflect, with_jamming,k):
        torch.save(self.actor.state_dict(),
                   model_path + f'/SAC_M{ele}_b{beta}_R{with_reflect}_J{with_jamming}_k{k}_actor.pth')
        torch.save(self.q_critic.state_dict(),
                   model_path + f'/SAC_M{ele}_b{beta}_R{with_reflect}_J{with_jamming}_k{k}_critic.pth')

    def load(self, model_path, ele, beta, with_reflect, with_jamming,k):
        self.actor.load_state_dict(
            torch.load(model_path + f'/SAC_M{ele}_b{beta}_R{with_reflect}_J{with_jamming}_k{k}_actor.pth',
                       map_location=self.dvc))
        self.q_critic.load_state_dict(
            torch.load(model_path + f'/SAC_M{ele}_b{beta}_R{with_reflect}_J{with_jamming}_k{k}_critic.pth',
                       map_location=self.dvc))


class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_size):
        super(Actor,self).__init__()

        # self.fc0 = nn.Linear(state_dim,512)
        self.fc1 = nn.Linear(state_dim,hidden_size[0])
        self.ln1= nn.LayerNorm(hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])



        self.mu = nn.Linear(hidden_size[1],action_dim)
        self.std = nn.Linear(hidden_size[1],action_dim)

        self.LOG_STD_MAX = 2
        # print(self.LOG_STD_MAX)
        self.LOG_STD_MIN = -20

    def forward(self,state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        # a =torch.tanh(self.fc0(state))
        a = torch.tanh(self.fc1(state))
        a = (self.ln1(a))
        a = torch.tanh(self.fc2(a))
        a = (self.ln2(a))
        mu = self.mu(a)
        log_std = self.std(a)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic: u = mu
        else: u = dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

class Double_Q_Critic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_size):
        super(Double_Q_Critic,self).__init__()



        self.fc1 = nn.Linear(state_dim +action_dim,hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1],1)


        self.fc4 = nn.Linear(state_dim +action_dim, hidden_size[0])
        self.fc5 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc6 = nn.Linear(hidden_size[1], 1)

    def forward(self,state ,action):

        sa = torch.cat([state, action], 1)


        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)


        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2





class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        # 每次只放入一个时刻的数据
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)  # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size  # 存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]















