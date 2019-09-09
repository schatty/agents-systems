import gym
import numpy as np
import pandas as pd


class EnvWrapper:
    def __init__(self, config):
        self.env = None
        self.config = config

        data_normalization_path = config['data_normalization_path']
        if data_normalization_path is not None:
            reward_data = pd.read_csv(f"{data_normalization_path}/stats_reward.csv")
            self.reward_min = float(reward_data['reward_min'][0])
            self.reward_max = float(reward_data['reward_max'][0])
            obs_data = pd.read_csv(f"{data_normalization_path}/stats_obs.csv")
            self.obs_min = np.array(obs_data['obs_min'])
            self.obs_max = np.array(obs_data['obs_max'])
        else:
            self.obs_mu = np.ones(config['state_dims'])
            self.obs_sigma = np.ones(config['state_dims'])
            self.reward_mu = 0
            self.reward_sigma = 1

    def reset(self):
        state = self.env.reset()
        state = self.normalize_state(state)
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        next_state = self.normalize_state(next_state)
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
        return (state - self.obs_min) / (self.obs_max - self.obs_min)

    def normalize_reward(self, reward):
        return (reward - self.reward_min) / (self.reward_min - self.reward_max)
