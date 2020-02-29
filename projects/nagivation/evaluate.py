import argparse
import numpy as np
from unityagents import UnityEnvironment

from env import UnityEnvWrapper
from models.agent import Agent


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
            if i == 0:
                img = env.render()
                print("Render: ", type(img))
                print(img.shape)
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
    parser.add_argument("--config", help="Path to the config file.")
    parser.add_argument("--n", default=10,
                        help="Number of times to evaluate model.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    path = "/home/igor/Banana_Linux/Banana"
    env = UnityEnvWrapper(UnityEnvironment(file_name=path))
    env.reset()

    agent = Agent(state_size=env.state_dim, action_size=env.action_dim, seed=0)
    agent.load("saved_models/model")

    eval_score = evaluate(env, agent, n_episodes=args.n)
    print(f"Eval score: {eval_score:5.3f}")
