import time
import os
import argparse
import numpy as np
from collections import deque
from unityagents import UnityEnvironment

import torch

from env import UnityEnvWrapper
from models.dqn import DQN
from utils import load_config


class Trainer:
    """Runs train procedure from config. """

    def __init__(self, config):
        self.config = config

    def train(self):
        config = self.config
        torch.manual_seed(config["seed"])

        env = UnityEnvWrapper(UnityEnvironment(file_name=config["env_path"]))
        env.reset()

        agent = DQN(config, env.state_dim, env.action_dim)

        # Epsilon parameters
        eps_start = config["eps_start"]
        eps_end = config["eps_end"]
        eps_decay = config["eps_decay"]

        scores = []
        scores_window = deque(maxlen=100)
        eps = eps_start
        time_start = time.time()
        for i_ep in range(1, config["n_episodes"]+1):
            state = env.reset()
            score = 0
            while True:
                action = agent.act(state, eps)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay*eps)
            mean_score = np.mean(scores_window)
            print(f'\rEpisode {i_ep}\tAverage Score: {mean_score:.2f}', end="")
            if i_ep % 100 == 0:
                print(f'\rEpisode {i_ep}\tAverage Score: {mean_score:.2f}')
                agent.save("saved_models/model")

        time_elapsed = time.time() - time_start()
        print(f"Traing took: {time_elapsed // 3600:4.3d} hours")

        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml",
                        help="Path to the .yaml config.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed of higher priority then in config.")
    args = parser.parse_args()
    return args


def save_scores(scores, save_dir, model_name, seed):
    fn = f"{save_dir}/{model_name}-{seed}.npy"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(fn, "wb") as f:
        np.save(f, scores)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Seed from arguments has higher priority
    if args.seed != -1:
        config["seed"] = args.seed

    trainer = Trainer(config)
    scores = trainer.train()
    save_scores(scores, config["results_dir"], config["model"], config["seed"])
