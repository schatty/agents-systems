from datetime import datetime
from multiprocessing import set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import os
from shutil import copyfile

from .ddpg import LearnerD3PG
from .agent import Agent
from utils.misc import read_config, empty_torch_queue
from utils.logger import Logger
from .replay_buffer import ReplayBuffer
from .networks import PolicyNetwork


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
    num_steps_train = config['num_steps_train']
    log_every = [1, config['num_steps_train'] // 1000][config['num_steps_train'] > 1000]

    # Logger
    log_dir = f"{log_dir}/data_struct"
    logger = Logger(log_dir)

    # Create replay buffer
    replay_buffer = ReplayBuffer(size=int(1e6))

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

        if update_step.value % 1000 == 0:
            print("Step: ", update_step.value, " buffer: ", len(replay_buffer))

        # Log data structures sizes
        if update_step.value % log_every == 0:
            step = global_episode.value
            logger.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
            logger.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
            logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    if config['analyze_replay_buffer']:
        replay_buffer.upload_stats(log_dir)

    empty_torch_queue(batch_queue)
    print("Replay buffer final size: ", len(replay_buffer))
    print("Stop sampler worker.")


def agent_worker(config, policy, global_episode, n_agent, log_dir, training_on, replay_queue, update_step, agent_type):
    agent = Agent(config,
                  policy,
                  global_episode=global_episode,
                  n_agent=n_agent,
                  log_dir=log_dir,
                  agent_type=agent_type)
    agent.run(training_on, replay_queue, update_step)


def learner_worker(config, local_policy, target_policy, log_dir, training_on, batch_queue, update_step):
    learner = LearnerD3PG(config, local_policy, target_policy, log_dir=log_dir)
    learner.run(training_on, batch_queue, update_step)


class ExperimentEngine(object):
    def __init__(self, config_path):
        self.config = read_config(config_path)

        # Create directory for experiment
        self.experiment_dir = f"{self.config['results_path']}/{self.config['env']}-{self.config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if config_path is not None:
            copyfile(config_path, f"{self.experiment_dir}/config.yml")

    def run(self):
        config = self.config

        replay_queue_size = config['replay_queue_size']
        batch_queue_size = config['batch_queue_size']
        n_agents = config['num_agents']
        n_exploiters = config['num_exploiters']

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=replay_queue_size)
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        training_on = torch_mp.Value('i', 1)
        update_step = torch_mp.Value('i', 0)
        global_episode = torch_mp.Value('i', 0)

        # Data sampler
        p = torch_mp.Process(target=sampler_worker,
                             args=(config, replay_queue, batch_queue, training_on,
                                   global_episode, update_step, self.experiment_dir))
        processes.append(p)

        # Learner (neural net training process)
        local_policy = PolicyNetwork(config["state_dims"],
                                     config["action_dims"],
                                     config["dense_size"],
                                     activation=config['policy_output_nonlinearity'],
                                     device=config["device"])
        local_policy.share_memory()
        target_policy = PolicyNetwork(config["state_dims"],
                                     config["action_dims"],
                                     config["dense_size"],
                                     activation=config['policy_output_nonlinearity'],
                                     device=config["device"])
        target_policy.share_memory()
        p = torch_mp.Process(target=learner_worker,
                             args=(config, local_policy, target_policy, self.experiment_dir, training_on, batch_queue, update_step))
        processes.append(p)

        # Agents (exploitation processes)
        for i in range(n_exploiters):
            p = torch_mp.Process(target=agent_worker,
                                 args=(self.config, target_policy, global_episode, i,
                                       self.experiment_dir, training_on, replay_queue, update_step, "exploitation"))
            processes.append(p)

        # Agents (exploration processes)
        for i in range(n_exploiters, n_exploiters+n_agents):
            p = torch_mp.Process(target=agent_worker,
                                 args=(self.config, local_policy, global_episode, i,
                                       self.experiment_dir, training_on, replay_queue, update_step, "exploration"))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print("End.")
