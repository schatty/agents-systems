import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


import gym


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()

        hidden_size = config["hidden_size"]
        state_dims = config["state_dims"]
        action_dims = config["action_dims"]

        self.affine = nn.Linear(state_dims, hidden_size)
        self.action_layer = nn.Linear(hidden_size, action_dims)
        self.value_layer = nn.Linear(hidden_size, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calc_loss(self, gamma=0.99):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)

        return loss

    def clear_memory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


class ActorCriticAgent(object):
    def __init__(self, config):
        self.config = config

        self.n_episodes = config["episodes"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.betas = (config["beta_start"], config["beta_end"])
        self.max_timestep = config["max_timestep"]
        #self.worker_processes = config['num_workers']

        self.actor_critic = ActorCritic(config)

    def run_n_episodes(self):
        env = gym.make("LunarLander-v2")

        policy = ActorCritic(self.config)
        optimizer = optim.Adam(policy.parameters(), lr=self.lr, betas=self.betas)

        episode_rewards = []
        running_reward = 0
        for i_episode in range(self.n_episodes):
            state = env.reset()
            episode_reward = 0
            for t in range(self.max_timestep):
                action = policy(state)
                state, reward, done, _ = env.step(action)
                policy.rewards.append(reward)
                running_reward += reward
                episode_reward += reward
                if done:
                    break

            episode_rewards.append(episode_reward)

            # Updating the policy
            optimizer.zero_grad()
            loss = policy.calc_loss(self.gamma)
            loss.backward()
            optimizer.step()
            policy.clear_memory()

            if i_episode % 20 == 0:
                running_reward = running_reward / 20
                print(f"Episode {i_episode}\tlength: {t}\treward {running_reward}")
                running_reward = 0

        return episode_rewards