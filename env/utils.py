import gym
from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker
from .learn_to_move import LearnToMove


def create_env_wrapper(config):
    env = config['env'].lower()
    if env == "pendulum-v0":
        return PendulumWrapper(config)
    elif env == "bipedalwalker-v2":
        return BipedalWalker(config)
    elif env == 'learntomove':
        return LearnToMove(config)
    else:
        raise ValueError("Unknown environment.")