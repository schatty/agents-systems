import gym
import numpy as np


class CientToEnv:
    def __init__(self, client):
        """
        Wrapper that reformats client environment to a local environment format,
        compelete with observation_space, reset, step, submit and time_limit.
        """
        self.client = client
        self.reset = client.env_reset
        self.step = client.env_step
        self.submit = client.submit
        self.time_limit = 300
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(22,),
                dtype=np.float32)
