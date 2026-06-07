import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self,env,**kwargs):
        self.__dict__.update(kwargs)

        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.BS_ant_num = env.BS_ant_num
        self.user_num = env.user_num
        self.RIS_ant_num = env.RIS_ant_num
        self.attacker_num = env.attacker_num
        self.MaxAction = np.ones(self.action_dim).reshape(1, -1)

        # self.writer = SummaryWriter('./td3')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.MaxAction = torch.FloatTensor(np.array(self.MaxAction)).to(self.device)
        hiddenside = [self.state_dim*2, self.state_dim ]

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

        self.action_dim = self.action_dim

        self.gamma = self.gamma
        self.policy_noise = 0.05
        self.noise_clip = 0.5
        self.tau = 0.005
        self.batch_size = self.batch_size
        self.delay_counter = -1
        self.delay_freq = 1
        self.q_iteration = 0
        self.a_iteration = 0

    def select_action(self, state):  # only used when interacting with the environment
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
            a = self.actor(state)
            # print("aaa",a)
        return a.cpu().numpy().flatten()


    def Update(self, batch_size):
        self.delay_counter += 1
        s, a, r, s_prime, done = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():

            noise = torch.max(-self.MaxAction, torch.min(torch.randn_like(a) * self.policy_noise, self.MaxAction))
            smoothed_target_a = torch.max(-self.MaxAction, torch.min(self.actor_target(s_prime) + noise, self.MaxAction))
        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
        target_Q = torch.min(target_Q1, target_Q2)

        target_Q = r + self.gamma * target_Q  # env without dead


        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()


        if self.delay_counter > self.delay_freq:
            # Update Actor

            a_pred = self.actor(s)

            Q1,_ = self.q_critic(s, a_pred)
            a_loss = -Q1.mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()


            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = 0

    def save(self,model_path,ele,beta,p,ant,error,with_jamming,pj ):
        torch.save(self.actor.state_dict(), model_path+f'/TD3_N{ant}_M{ele}_b{beta}_P{p}_e{error}_J{with_jamming}_pj{pj}_actor.pth')
        torch.save(self.q_critic.state_dict(), model_path+f'/TD3_N{ant}_M{ele}_b{beta}_P{p}_e{error}_J{with_jamming}_pj{pj}_critic.pth')

    def load(self,model_path, ele,beta, p,ant,error,with_jamming,pj):
        self.actor.load_state_dict(
            torch.load(model_path+f'/TD3_N{ant}_M{ele}_b{beta}_P{p}_e{error}_J{with_jamming}_pj{pj}_actor.pth', map_location=self.dvc))
        self.q_critic.load_state_dict(
            torch.load(model_path+f'/TD3_N{ant}_M{ele}_b{beta}_P{p}_e{error}_J{with_jamming}_pj{pj}_critic.pth', map_location=self.dvc))



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim , hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], action_dim)

        self.ln1 = nn.LayerNorm(hidden_size[0])  # Layer Normalization
        self.ln2 = nn.LayerNorm(hidden_size[1])  # Layer Normalization

    def forward(self, state):

        a = torch.tanh(self.l1(state))
        # a = self.ln1(a)
        a = torch.tanh(self.l2(a))
        # a = self.ln2(a)
        a = torch.tanh(self.l3(a))
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Q_Critic, self).__init__()



        self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l5 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l6 = nn.Linear(hidden_size[1], 1)


    def forward(self, state, action):

        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
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




