import numpy as np
from .env_wrapper import EnvWrapper


class PendulumWrapper(EnvWrapper):
    def __init__(self, config):
        EnvWrapper.__init__(self, config['env'])
        self.config = config

    def normalise_state(self, state):
        return state

    def step(self, action):
        action *= 2 # Multiply output of tanh with 2 as Pendulum action range is [-2, 2]
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        return next_state, reward, terminal

    def normalise_reward(self, reward):
        return reward/100.0

