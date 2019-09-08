import os
from time import time
import numpy as np
import pandas as pd
import datetime
import itertools

import torch
import gym

from dqn import run_training as run_dqn_learning


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.reset()

    configs = {
        'env': [env],
        'trials': [1000],
        'trial_len': [500],
        'gamma': [0.85, 0.9],
        'epsilon': [1.0],
        'epsilon_min': [0.01],
        'epsilon_decay': [0.995, 0.999],
        'learning_rate': [0.005],
        'hidden_sizes': [[20, 20, 20], [30, 60, 30], [20, 50, 20]],
        'tau': [0.1, 0.125],
        'mem_size': [1000, 2000, 3000],
        'memory_batch': [32, 64],
        'cuda': [True],
        'gpu_num': [0],
        'seed': [2019]
    }

    # Create grid of parameters
    keys, values = zip(*configs.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create resulting file if necessary
    now = datetime.datetime.now()
    cur_date = now.strftime("%Y-%m-%d")
    results_dir = f"MountainCar-v0/results/{cur_date}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    res_path = f"{results_dir}/dqn_learing.csv"
    if not os.path.exists(res_path):
        df = pd.DataFrame(columns=configs.keys())
        df.to_csv(res_path, index=False)

    conf_durations = []
    for i, param in enumerate(param_grid):
        if len(conf_durations):
            time_estimate = (len(param_grid) - (i+1)) * np.mean(conf_durations) // 60
        else:
            time_estimate = '-'
        print(f"Configuration: ", param)
        print(f"Progress {i+1}/{len(param_grid)}. Estimated time until end: {time_estimate} min")
        time_start = time()
        rewards = run_dqn_learning(config=param)
        conf_durations.append(time() - time_start)
        df = pd.read_csv(res_path)
        df = df.append(pd.Series({**param, **{'mean_rewards': np.mean(rewards),
                                              'last_reward': rewards[-1],
                                              'duration_sec': conf_durations[-1]}}), ignore_index=True)
        df.to_csv(res_path, index=False)

