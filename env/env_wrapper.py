import gym
import numpy as np
import pandas as pd
import pickle


class EnvWrapper:
    def __init__(self, config):
        self.env = None
        self.config = config

        data_normalization_path = config['data_normalization_path']
        if data_normalization_path is not None:
            with open(config['data_normalization_path'], 'rb') as f:
                data = pickle.load(f)
                self.obs_min = data['min']
                self.obs_max = data['max']
        else:
            self.obs_min = 0
            self.obs_max = 1

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
        #return (reward - self.reward_min) / (self.reward_min - self.reward_max)
        return reward