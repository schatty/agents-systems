import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dense_size, device="cuda"):
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.l1 = nn.Linear(state_dim, dense_size)
        self.l2 = nn.Linear(dense_size, dense_size)
        self.l3 = nn.Linear(dense_size, action_dim)

        self.max_action = max_action
        self.to(device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dense_size):
        super(ValueNetwork, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, dense_size)
        self.l2 = nn.Linear(dense_size, dense_size)
        self.l3 = nn.Linear(dense_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, dense_size)
        self.l5 = nn.Linear(dense_size, dense_size)
        self.l6 = nn.Linear(dense_size, 1)

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