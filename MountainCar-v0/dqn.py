import gym
import numpy as np
import random
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_dim, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class DQN:
    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, tau, mem_size, memory_batch, device):
        self.env = env
        self.memory = deque(maxlen=mem_size)
        self.memory_batch = memory_batch

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n

        self.model = MLP(in_dim, out_dim)
        self.target_model = MLP(in_dim, out_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.model.to(device)
        self.target_model.to(device)
        self.device = device

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        st = self.model(state).cpu().detach().numpy()
        return np.argmax(st[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        self.model.train()
        if len(self.memory) < self.memory_batch:
            return

        samples = random.sample(self.memory, self.memory_batch)
        for sample in samples:
            state, action, reward, new_state, done = sample

            state = torch.tensor(state).float().to(self.device)
            new_state = torch.tensor(state).float().to(self.device)

            target = self.target_model(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            target_ = self.model(state)
            loss = self.criterion(target_, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_train(self):
        self.target_model.fc1.weight.data.copy_(self.model.fc1.weight.data * self.tau + self.target_model.fc1.weight.data * (1-self.tau))
        self.target_model.fc1.bias.data.copy_(self.model.fc1.bias.data * self.tau + self.target_model.fc1.bias.data * (1-self.tau))
        self.target_model.fc2.weight.data.copy_(self.model.fc2.weight.data * self.tau + self.target_model.fc2.weight.data * (1-self.tau))
        self.target_model.fc2.bias.data.copy_(self.model.fc2.bias.data * self.tau + self.target_model.fc2.bias.data * (1-self.tau))
        self.target_model.fc3.weight.data.copy_(self.model.fc3.weight.data * self.tau + self.target_model.fc3.weight.data * (1-self.tau))
        self.target_model.fc3.bias.data.copy_(self.model.fc3.bias.data * self.tau + self.target_model.fc3.bias.data * (1-self.tau))
        self.target_model.fc4.weight.data.copy_(self.model.fc4.weight.data * self.tau + self.target_model.fc4.weight.data * (1-self.tau))
        self.target_model.fc4.bias.data.copy_(self.model.fc4.bias.data * self.tau + self.target_model.fc4.bias.data * (1-self.tau))


    def save_model(self, fn):
        # TODO: Add saving logic
        pass


def run_training(config):
    # Reproducibility
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Set device on which to train
    cuda_on = torch.cuda.is_available() and config['cuda']
    gpu_num = config['gpu_num']
    device = torch.device(f"cuda:{gpu_num}" if cuda_on else "cpu")

    # Num of trials
    env = config['env']
    trials = config['trials']
    trial_len = config['trial_len']

    # Agent as DQN
    agent = DQN(env=env,
                gamma=config['gamma'],
                epsilon=config['epsilon'],
                epsilon_min=config['epsilon_min'],
                epsilon_decay = config['epsilon_decay'],
                learning_rate=config['learning_rate'],
                tau=config['tau'],
                mem_size=config['mem_size'],
                memory_batch=config['memory_batch'],
                device=device)

    for trial in tqdm(range(trials)):
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            cur_state = torch.tensor(cur_state).float().to(device)

            # Agent Acts
            action = agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 2)

            # Update agent's memory
            agent.remember(cur_state, action, reward, new_state, done)

            # Learn from memory
            agent.replay()
            # Train target network
            agent.target_train()

            cur_state = new_state
            if done:
                break
        if step < 199:
            print(f"Completed on the {trial} trial")
            agent.save_model("dqn.pt")
            break


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.reset()

    config = {
        'env': env,
        'trials': 1000,
        'trial_len': 500,
        'gamma': 0.85,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 0.005,
        'tau': 125,
        'mem_size': 3000,
        'memory_batch': 32,
        'cuda': True,
        'gpu_num': 0,
        'seed': 2019
    }

    run_training(config)