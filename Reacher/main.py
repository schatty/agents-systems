from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from agent import Agent


def main():
    fn = "Reacher_Linux_NoVis/Reacher.x86_64"
    env = UnityEnvironment(file_name=fn, worker_id=1)

    # Get default brain 
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Number of agents
    num_agents = len(env_info.agents)
    print("Number of agents: ", num_agents)

    # Size of each action
    action_size = brain.vector_action_space_size
    print("Size of each action: ", action_size)

    # Size of each state
    state_size = brain.vector_observation_space_size
    print("Size of each state: ", state_size)


if __name__ == "__main__":
    main()
