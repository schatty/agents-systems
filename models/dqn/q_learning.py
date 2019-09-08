import time
import os
import numpy as np
import pandas as pd
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib
matplotlib.use('Agg')
plt.figure(figsize=(9,9))


def run_training(config):
    """
    Run training process of MountainCar environment
    Args:
        config (dict): full configuration of experiment

    Returns (np.ndarray): array of rewards

    """
    np.random.seed(config['seed'])
    env = config['env']
    learning_rate = config['lr']
    discount = config['discount']
    epsilon = config['epsilon']
    min_eps = config['min_eps']
    episodes = config['episodes']
    monitor_reward_eps = config['monitor_reward_eps']
    display = config['display']
    img_folder = config['img_folder']
    states_size = np.array(config['states_size'])

    # Determine size of discritezed state space
    num_states = (env.observation_space.high - env.observation_space.low) * states_size
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1, size=(
    num_states[0], num_states[1], env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    avg_reward_list = []

    # Calculate episodic reduction in epsilon
    eps_reduction = (epsilon - min_eps) / episodes

    # Create folder for the images
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Run Q-learning algorithm
    for i_episode in tqdm(range(episodes)):
        reward_total, reward_step = 0, 0
        state = env.reset()

        # Discretize space
        state_discrete = (state - env.observation_space.low) * states_size
        state_discrete = np.round(state_discrete, 0).astype(int)

        step = 0
        while True:
            step += 1
            # Render environment for last episode
            if display and i_episode == episodes - 1:
                plt.imshow(env.render(mode='rgb_array'))
                plt.savefig(f'{img_folder}/{step}.png')

            # Epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_discrete[0], state_discrete[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            next_state, reward_step, done, info = env.step(action)

            # Discretize next state
            next_state_discrete = (next_state - env.observation_space.low) * states_size
            next_state_discrete = np.round(next_state_discrete, 0).astype(int)

            # Terminal state
            if done and next_state[0] >= 0.5:
                Q[state_discrete[0], state_discrete[1], action] = reward_step
            # Update Q-value
            else:
                delta = learning_rate * (reward_step + discount * np.max(
                    Q[next_state_discrete[0], next_state_discrete[1]]) - Q[
                                             state_discrete[0], state_discrete[
                                                 1], action])
                Q[state_discrete[0], state_discrete[1], action] += delta

            reward_total += reward_step
            state_discrete = next_state_discrete

            if done: break

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= eps_reduction

        # Track rewards
        reward_list.append(reward_total)

        if (i_episode + 1) % monitor_reward_eps == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []

    env.close()

    return avg_reward_list


def plot_reward(rewards, output_file):
    """
    Plot rewards during training and save it to the file.

    Args:
        rewards (list): list of float rewards
        output_file (str): name of the output file

    Returns: None

    """
    plt.figure(figsize=(6, 3), dpi=100)
    data = np.array([100 * (np.arange(len(rewards)) + 1), rewards]).T
    df = pd.DataFrame(data, columns=['Episodes', 'Rewards'])
    plt.title("Average Reward vs Episode")
    sns.lineplot(data=df, x='Episodes', y='Rewards')
    plt.savefig(output_file)


if __name__ == "__main__":
    # Import environment
    env = gym.make("MountainCar-v0")
    env.reset()

    # Experiment configuration
    config = {
        'env': env,
        'lr': 0.2,
        'discount': 0.9,
        'epsilon': 0.8,
        'min_eps': 0,
        'states_size': [15, 80],
        'episodes': 10_000,
        'monitor_reward_eps': 10,
        'display': False,
        'img_folder': 'imgs',
        'seed': 2019
    }

    # Run Q-learning algorithm
    rewards = run_training(config)
    print("Mean reward: ", np.mean(rewards))
    print("Last reward: ", rewards[-1])

    # Plot rewards
    plot_reward(rewards, "reward.png")