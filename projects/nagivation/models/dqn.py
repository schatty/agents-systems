import os
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=64):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Hidden size
        """
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        o = self.relu(self.linear1(state))
        o = self.relu(self.linear2(o))
        o = self.linear3(o)
        return o


class DQN():
    def __init__(self, config, state_size, action_size):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = config["batch_size"]
        self.device = config["device"]
        self.config = config

        # Q-Network
        self.qnet_local = QNetwork(state_size, action_size).to(self.device)
        self.qnet_target = QNetwork(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.qnet_local.parameters(),
                                    lr=config["lr"])

        # Replay memory
        self.memory = ReplayBuffer(action_size,
                                   config["buffer_size"],
                                   config["batch_size"],
                                   config["seed"],
                                   config["device"])
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.config["update_every"]
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                ss = self.memory.sample()
                self._update_step(ss, self.config["gamma"])

    def _update_step(self, ss, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            ss (tuple): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = ss

        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        next_states = next_states.cuda()
        dones = dones.cuda()

        q_values = self.qnet_local(states)
        q_values_next = self.qnet_target(next_states)

        q_value = q_values.gather(1, actions)  # [64, 1]
        q_value_next = q_values_next.max(1)[0]
        q_value_next = q_value_next.unsqueeze(1)  # [64, 1]

        expected_q_value = rewards + gamma * q_value_next * (1 - dones)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnet_local, self.qnet_target,
                         self.config["tau"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters. """
        for target_p, local_p in zip(target_model.parameters(),
                                     local_model.parameters()):
            target_p.data.copy_(tau*local_p.data + (1.0-tau)*target_p.data)

    def save(self, save_dir):
        """Save model to the given directory.
        Args:
            save_dir (str): directory to save the model.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.qnet_local, f"{save_dir}/qnet_local.pt")
        torch.save(self.qnet_target, f"{save_dir}/qnet_target.pt")

    def load(self, load_dir):
        self.qnet_local = torch.load(f"{load_dir}/qnet_local.pt")
        self.qnet_target = torch.load(f"{load_dir}/qnet_target.pt")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        field_names = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=field_names)
        self.seed = seed
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def _to_cuda(self, l, ttype=None):
        t = np.vstack(l)
        if ttype is not None:
            t = t.astype(ttype)
        t = torch.from_numpy(t).float().to(self.device)
        return t

    def sample(self):
        """Randomly sample a batch of ss from memory."""
        ss = random.sample(self.memory, k=self.batch_size)

        states = self._to_cuda([e.state for e in ss if e is not None])
        actions = self._to_cuda([e.action for e in ss if e is not None]).long()
        rewards = self._to_cuda([e.reward for e in ss if e is not None])
        next_states = self._to_cuda([e.next_state
                                     for e in ss if e is not None])
        dones = self._to_cuda([e.done for e in ss if e is not None], float)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
