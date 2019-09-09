import gym
from .env_wrapper import EnvWrapper


class BipedalWalker(EnvWrapper):
    def __init__(self, config):
        EnvWrapper.__init__(self, config)
        self.env_name = config['env']
        self.env = gym.make(self.env_name)

