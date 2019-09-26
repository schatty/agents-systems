import numpy as np
import matplotlib
# MacOS issues
matplotlib.use('PS')
from osim.env import L2M2019Env
import torch

from env.learn_to_move import ObservationTransformer

# Create environment
env = L2M2019Env(visualize=True)
env.reset()
observation = env.reset()

controller = torch.load("data/models/policy_network.pt", map_location=torch.device('cpu'))
controller.device = 'cpu'

obs_transformer = ObservationTransformer()
print("Controller loaded.")

n_trials = 10
rewards = []
total_reward = 0
while True:
    observation = obs_transformer.transform(observation)
    action = controller.get_action(observation).flatten().tolist()
    [observation, reward, done, info] = env.step(action)

    #print("Pelvis: ", observation['pelvis'].keys())
    #print("pelvis pitch: ", observation['pelvis']['pitch'])
    #print("pelvis roll: ", observation['pelvis']['roll'])
    #print("Observation: ", observation['r_leg']['joint'])
    #print("joint_d: ", observation['r_leg']['d_joint'])

    total_reward += reward
    if done:
        print(f"Reward: {total_reward}")
        rewards.append(total_reward)
        total_reward = 0
        observation = env.reset()
        if not observation or len(rewards) == n_trials:
            break
print(f"Mean reward from {n_trials} trials: {np.mean(rewards)}")