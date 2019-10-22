import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dense_size):
        super(PolicyNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, dense_size)
        self.l2 = nn.Linear(dense_size, dense_size)
        self.l3 = nn.Linear(dense_size, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dense_size, v_min=0, v_max=100, num_atoms=50):
        super(ValueNetwork, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, dense_size)
        self.l2 = nn.Linear(dense_size, dense_size)
        self.l3 = nn.Linear(dense_size, num_atoms)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, dense_size)
        self.l5 = nn.Linear(dense_size, dense_size)
        self.l6 = nn.Linear(dense_size, num_atoms)

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def get_probs(self, state, action):
        q1, q2 = self.forward(state, action)
        return F.softmax(q1, dim=1), F.softmax(q2, dim=1)

    def get_probs_q1(self, state, action):
        return F.softmax(self.Q1(state, action), dim=1)