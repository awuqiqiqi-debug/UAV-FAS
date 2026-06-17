import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions, name,
                 chkpt_dir, suffix='_TD3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + suffix)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc4_dims)
        f5 = 0.003
        self.q = nn.Linear(self.fc4_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f5, f5)
        T.nn.init.uniform_(self.q.bias.data, -f5, f5)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = self.bn4(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        print('... loading checkpoint ...')
        if T.cuda.is_available():
            self.load_state_dict(T.load(load_file))
        else:
            self.load_state_dict(T.load(load_file, map_location=T.device('cpu')))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions, name,
                 chkpt_dir, suffix='_TD3', n_discrete_dims=0):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.n_discrete_dims = n_discrete_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name + suffix)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        self.fc4.weight.data.uniform_(-f4, f4)
        self.fc4.bias.data.uniform_(-f4, f4)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        f5 = 0.003
        if self.n_discrete_dims > 0:
            self.mu_discrete = nn.Linear(self.fc4_dims, self.n_discrete_dims)
            self.mu_discrete.weight.data.uniform_(-f5, f5)
            self.mu_discrete.bias.data.uniform_(-f5, f5)
            n_continuous = self.n_actions - self.n_discrete_dims
            self.mu_continuous = nn.Linear(self.fc4_dims, n_continuous)
            self.mu_continuous.weight.data.uniform_(-f5, f5)
            self.mu_continuous.bias.data.uniform_(-f5, f5)
        else:
            self.mu = nn.Linear(self.fc4_dims, self.n_actions)
            self.mu.weight.data.uniform_(-f5, f5)
            self.mu.bias.data.uniform_(-f5, f5)

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

        if self.n_discrete_dims > 0:
            logits = self.mu_discrete(x)
            continuous = T.tanh(self.mu_continuous(x))
            return T.cat([logits, continuous], dim=-1)
        else:
            return T.tanh(self.mu(x))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        print('... loading checkpoint ...')
        if T.cuda.is_available():
            state_dict = T.load(load_file)
        else:
            state_dict = T.load(load_file, map_location=T.device('cpu'))
        # strict=False: 允许新旧架构不完全匹配 (mu→mu_discrete/mu_continuous)
        self.load_state_dict(state_dict, strict=False)
