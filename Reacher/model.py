import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetwork(nn.Module):

    def __init__(self, state_size, action_size, output_gate=None):
        super(FCNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.output_gate = output_gate

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.output_gate is not None:
            x = self.output_gate(x)
        return x


class ActorCriticNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()

        self.actor = FCNetwork(state_size, action_size, F.tanh)
        self.critic = FCNetwork(state_size, 1)
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.std = nn.Parameter(torch.ones(1, action_size)).to(self.deivce)
        self.to(self.device)

    def forward(self, state, action=None):
        # Get action
        a = self.actor(state)
        distribution = torch.distributions.Normal(a, self.std)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        # Get value from critic
        value = self.critic(state)

        return action, log_prob, value
