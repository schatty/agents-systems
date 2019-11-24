import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
sns.set_style('whitegrid')
colors = ['greyish', 'faded blue', 'pink', "faded green", 'olive']
sns.set_palette(sns.xkcd_palette(colors))


def load_xy(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    data = np.asarray(data)
    return data[:, 1:]


def plot_data(paths, output_path, n_rows=1, n_cols=3, smooth_len=5, lw=3, figsize=(20, 5)):
    figure(num=0, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')
    for i_subplot, env_name in enumerate(paths):
        ax = plt.subplot(n_rows, n_cols, i_subplot + 1)
        plt.title(env_name)

        for model_name in paths[env_name]:
            data = load_xy(paths[env_name][model_name])

            # Smooth reward
            for i in range(data.shape[0] - 1, smooth_len, -1):
                data[i, 1] = np.mean(data[i - smooth_len:i, 1])

            data = pd.DataFrame(data, columns=['step', 'reward'])
            data['model'] = model_name
            sns.lineplot(data=data, x='step', y='reward', lw=lw)

        ax.legend(labels=list(paths[env_name]), loc='lower right')

    plt.savefig(output_path)


if __name__ == "__main__":
    paths = {
        'Ant-v2': {
            'DDPG': 'results/ddpg/ant_ddpg.json',
            'D3PG': 'results/d3pg/ant_d3pg.json',
            'D4PG': 'results/d4pg/ant_d4pg.json',
            'TD3': 'results/td3/ant_td3.json',
            'TD4': 'results/td3_distributed/ant_td3_distr.json'
        },
        'HalfCheetah-v2': {
            'DDPG': 'results/ddpg//half_cheetah_ddpg.json',
            'D3PG': 'results/d3pg/half_cheetah_d3pg.json',
            'D4PG': 'results/d4pg/half_cheetah_d4pg.json',
            'TD3': 'results/td3/half_cheetah_td3.json',
            'TD4': 'results/td3_distributed/half_cheetah_td3_distr.json'
        },
        'Hopper-v2': {
            'DDPG': 'results/ddpg/hopper_ddpg.json',
            'D3PG': 'results/d3pg/hopper_d3pg.json',
            'D4PG': 'results/d4pg/hopper_d4pg.json',
            'TD3': 'results/td3/hopper_td3.json',
            'TD4': 'results/td3_distributed/hopper_td3_distr.json',
        },
        'InvertedPendulum-v2': {
            'DDPG': 'results/ddpg/inverted_pendulum_ddpg.json',
            'D3PG': 'results/d3pg/inverted_pendulum_d3pg.json',
            'D4PG': 'results/d4pg/inverted_pendulum_d4pg.json',
            'TD3': 'results/td3/inverted_pendulum_td3.json',
            'TD4': 'results/td3_distributed/inverted_pendulum_td3_distr.json',
        },
        'InvertedDoublePendulum-v2': {
            'D3PG': 'results/d3pg/inverted_double_pendulum_d3pg.json',
            'D4PG': 'results/d4pg/inverted_double_pendulum_d4pg.json',
            'TD3': 'results/td3/inverted_double_pendulum_td3.json',
            'TD4': 'results/td3_distributed/inverted_double_pendulum_td3_distr.json'
        },
        'Walker2d-v2': {
            'DDPG': 'results/ddpg/walker_ddpg.json',
            'D3PG': 'results/d3pg/walker2d_d3pg.json',
            'D4PG': 'results/d4pg/walker2d_d4pg.json',
            'TD3': 'results/td3/walker_td3.json',
            'TD4': 'results/td3_distributed/walker2d_td3_distr.json'
        }
    }
    plot_data(paths, "plot.png", n_rows=2, n_cols=3, smooth_len=10, figsize=(20, 10))