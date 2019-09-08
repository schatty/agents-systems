import gym
from .pendulum import PendulumWrapper


def create_env_wrapper(config):
    env = config['env'].lower()
    if env == "pendulum-v0":
        return PendulumWrapper(config)
    else:
        raise ValueError("Unknown environment.")