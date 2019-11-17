import copy
from datetime import datetime
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import os

from models.agent import Agent

from .td3 import LearnerTD3
from .networks import PolicyNetwork
from utils.misc import read_config, empty_torch_queue
from utils.logger import Logger
from .utils import ReplayBuffer


def sampler_worker(config, replay_queue, batch_queue, training_on,
                   global_episode, update_step, log_dir=''):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    Args:
        config:
        replay_queue:
        batch_queue:
        training_on:
        global_episode:
        log_dir:
    """
    batch_size = config['batch_size']

    # Logger
    log_dir = f"{log_dir}/data_struct"
    logger = Logger(log_dir=log_dir)

    # Create replay buffer
    replay_buffer = ReplayBuffer(state_dim=config["state_dim"], action_dim=config["action_dim"])

    while training_on.value or not replay_queue.empty():
        # (1) Transfer replays to global buffer
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if not training_on.value:
            # Repeat loop to wait until replay_queue will be empty
            continue
        if len(replay_buffer) < batch_size:
            continue

        if not batch_queue.full():
            batch = replay_buffer.sample(batch_size)
            batch_queue.put(batch)

        # Log data structures sizes
        if (update_step.value+1) % config["eval_freq"] == 0:
            step = global_episode.value
            logger.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
            logger.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
            logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    empty_torch_queue(batch_queue)
    print("Replay buffer final size: ", len(replay_buffer))
    print("Stop sampler worker.")


def agent_worker(config, policy, learner_w_queue, global_episode, n_agent, log_dir, training_on, replay_queue, update_step):
    agent = Agent(config,
                  policy,
                  global_episode=global_episode,
                  n_agent=n_agent,
                  log_dir=log_dir)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


def learner_worker(config, local_policy, target_policy, learner_w_queue, log_dir, training_on, batch_queue, update_step):
    learner = LearnerTD3(config, local_policy, target_policy, learner_w_queue, log_dir=log_dir)
    learner.run(training_on, batch_queue, update_step)


class Engine(object):
    def __init__(self, config):
        self.config = config

        # Create directory for experiment
        self.experiment_dir = f"{self.config['results_path']}/{self.config['env']}-{self.config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def train(self):
        config = self.config

        replay_queue_size = config['replay_queue_size']
        batch_queue_size = config['batch_queue_size']
        n_agents = config['num_agents']

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=replay_queue_size)
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        learner_w_queue = torch_mp.Queue(maxsize=n_agents)
        training_on = torch_mp.Value('i', 1)
        update_step = torch_mp.Value('i', 0)
        global_episode = torch_mp.Value('i', 0)

        # Data sampler
        p = torch_mp.Process(target=sampler_worker,
                             args=(config, replay_queue, batch_queue, training_on,
                                   global_episode, update_step, self.experiment_dir))
        processes.append(p)

        target_policy = PolicyNetwork(config["state_dim"], config["action_dim"], config["action_high"], config["dense_size"])
        target_policy.share_memory()
        target_policy.to(config["device"])
        local_policy = copy.deepcopy(target_policy)

        if config['agent_device'] == 'cpu':
            agent_policy = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'])
        else:
            local_policy.share_memory()
            agent_policy = local_policy

        p = torch_mp.Process(target=learner_worker,
                             args=(config, local_policy, target_policy, learner_w_queue,
                                   self.experiment_dir, training_on, batch_queue, update_step))
        processes.append(p)

        # Agents
        for i in range(n_agents):
            p = torch_mp.Process(target=agent_worker,
                                 args=(self.config, agent_policy, learner_w_queue, global_episode, i,
                                       self.experiment_dir, training_on, replay_queue, update_step))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print("End.")