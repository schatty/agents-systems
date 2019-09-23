import numpy as np
from torch import multiprocessing
from torch.multiprocessing import Queue

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(num_states, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_actions + 1)

    def forward(self, state):
        pass


class A3C(object):
    def __init__(self, config):
        self.config = config
        self.worker_processes = config['num_workers']

        self.actor_critic = ActorCritic(config)

    def run_n_episodes(self):
        results_queue = Queue()
        gradient_updates_queue = Queue()
        episode_number = multiprocessing.Value('i', 0)
        optimizer_lock = multiprocessing.Lock()
        episodes_per_process = int(self.config["episodes"] / self.worker_processes) + 1

        processes = []

        self.actor_critic.share_memory()
        self.actor_critic_optimizer.share_memory()

        return [np.random.random() for _ in range(100)]