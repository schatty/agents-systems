import gym
import numpy as np
import pandas as pd


class EnvWrapper:
    def __init__(self, config, data_normalization_path=None):
        self.env = None
        self.config = config
        if data_normalization_path is not None:
            reward_data = pd.read_csv(f"{data_normalization_path}/stats_reward.csv")
            self.reward_mu = float(reward_data['reward_mean'][0])
            self.reward_sigma = float(reward_data['reward_std'][0])
            obs_data = pd.read_csv(f"{data_normalization_path}/stats_obs.csv")
            self.obs_mu = np.array(obs_data['obs_mean'])
            self.obs_sigma = np.array(obs_data['obs_std'])
        else:
            self.obs_mu = 0
            self.obs_sigma = 1
            self.reward_mu = 0
            self.reward_sigma = 1

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        return next_state, (reward, reward), terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    def normalize_state(self, state):
        return (state - self.obs_mu) / self.obs_sigma

    def normalize_reward(self, reward):
        return (reward - self.reward_mu) / self.reward_sigma
