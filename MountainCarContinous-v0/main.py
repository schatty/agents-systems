import gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 24)
        self.fc3 = nn.Linear(24, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=10_000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.005
        self.tau = .125

        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.shape[0]

        self.model = MLP(in_dim, out_dim)
        self.target_model = MLP(in_dim, out_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.model.to(device)
        self.target_model.to(device)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        st = self.model(state)
        st = st.cpu().detach().numpy()

        return np.argmax(st[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        self.model.train()
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample

            state = torch.tensor(state).float().to(device)
            new_state = torch.tensor(state).float().to(device)

            state = torch.tensor(state).float()
            new_state = torch.tensor(new_state).float()

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


    def save_model(self, fn):
        #self.model.save(fn)
        pass


def main():
    env = gym.make("MountainCarContinuous-v0")
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 300

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            cur_state = torch.tensor(cur_state).float().to(device)

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step([action])

            # reward = reward if not done else -20
            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    main()