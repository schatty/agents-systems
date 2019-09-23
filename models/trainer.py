import numpy as np
import matplotlib.pyplot as plt

from models.a3c.a3c import A3C
from models.actor_critic.actor_critic import ActorCriticAgent


def create_agent(config):
    if config['model'] == "A3C":
        return A3C(config)
    elif config['model'] == "ActorCritic":
        return ActorCriticAgent(config)


class Trainer(object):
    """Runs training procedure of given agent. """

    def __init__(self, config):
        self.config = config

    def train(self):
        agent = create_agent(self.config)
        scores = agent.run_n_episodes()

        plt.plot(np.arange(len(scores)), scores)
        plt.show()