import argparse
import numpy as np
from unityagents import UnityEnvironment

from env import UnityEnvWrapper
from models.dqn import DQN
from utils import load_config


def evaluate(env, agent, n_episodes=10, save_video=False, video_dir="demo"):
    """Run evaluation of an agent.

    Args:
        env: environment.
        agent: agent.
        n_episode: number of evaluation to take the mean of
        save_video (bool): flag to save video
        video_dir (str): dictory for the video
    """
    env.eval_mode()
    ep_scores = []
    for i in range(n_episodes):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state
            if done:
                break

        ep_scores.append(score)

    print("Scores: ", ep_scores)
    return np.mean(ep_scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml",
                        help="Path to the config file.")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of times to evaluate model.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    path = "/Users/igor/Downloads/Banana.app"
    env = UnityEnvWrapper(UnityEnvironment(file_name=path))
    env.reset()

    config["device"] = "cpu"
    agent = DQN(config, state_size=env.state_dim, action_size=env.action_dim)
    agent.load("saved_models/model", "cpu")

    eval_score = evaluate(env, agent, n_episodes=args.n)
    print(f"Eval score: {eval_score:5.3f}")
