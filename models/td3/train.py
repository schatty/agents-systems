import numpy as np
import os
import gym
import torch
import matplotlib.pyplot as plt

from models.td3.td3 import TD3
from models.td3.utils import ReplayBuffer

config = {
    "policy_name": "TD3",
    "env_name": "BipedalWalker-v2",
    "seed": 2,
    "start_timesteps": 0,
    "max_timesteps": 1000,
    "expl_noise": 0.1,
    "batch_size": 256,
    "episodes": 100,
    "eval_freq": 99,
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.1,
    "noise_clip": 0.5,
    "policy_freq": 2
}

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(config["env_name"])

    # Set seeds
    env.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": config["discount"],
        "tau": config["tau"],
        "policy_noise": config["policy_noise"] * max_action,
        "noise_clip": config["noise_clip"] * max_action,
        "policy_freq": config["policy_freq"]
    }

    # Initialize policy
    policy = TD3(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, config["env_name"], config["seed"])]
    episode_num = 0
    episode_rewards = []

    for i_ep in range(config["episodes"]):
        print("Episode ", i_ep)
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0

        for t in range(int(config["max_timesteps"])):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < config["start_timesteps"]:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(state)) + np.random.normal(0, max_action * config["expl_noise"], size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if len(replay_buffer) > config["batch_size"]:
                policy.train(replay_buffer, config["batch_size"])

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                break

            # Evaluate episode
            #if (i_ep + 1) % config["eval_freq"] == 0:
            #    evaluations.append(eval_policy(policy, config["env_name"], config["seed"]))

    print("Episode rewards: ", episode_rewards)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.show()
    plt.savefig("plot.png")
