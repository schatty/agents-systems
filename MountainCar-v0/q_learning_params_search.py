import os
from time import time
import itertools
import numpy as np
import pandas as pd
import datetime

from q_learning import run_training as run_q_learning
import gym


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.reset()

    configs = {
        'env': [env],
        'lr': [0.1, 0.2, 0.25, 0.3],
        'discount': [0.9, 0.99],
        'epsilon': [0.8, 0.9, 0.99],
        'min_eps': [0, 0.1],
        'states_size': [[5, 50], [10, 100], [15, 80], [10, 200]],
        'episodes': [5_000],
        'monitor_reward_eps': [10],
        'display': [False],
        'img_folder': ['imgs'],
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
    res_path = f"{results_dir}/q_learning.csv"
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
        rewards = run_q_learning(config=param)
        conf_durations.append(time() - time_start)
        df = pd.read_csv(res_path)
        df = df.append(pd.Series({**param, **{'mean_rewards': np.mean(rewards),
                                              'last_reward': rewards[-1],
                                              'duration_sec': conf_durations[-1]}}), ignore_index=True)
        df.to_csv(res_path, index=False)

