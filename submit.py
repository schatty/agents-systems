import opensim as osim
from osim.http.client import Client
import numpy as np
import torch

from env.learn_to_move import ObservationTransformer

# Settings
remote_base = "http://osim-rl-grader.aicrowd.com/"
aicrowd_token = "902addd6f5fcaf1de173fcbe0963ef30" # use your aicrowd token
# your aicrowd token (API KEY) can be found at your prorfile page at https://www.aicrowd.com

client = Client(remote_base)

# Create environment
observation = client.env_create(aicrowd_token, env_id='L2M2019Env')

# IMPLEMENTATION OF YOUR CONTROLLER
controller = torch.load("data/models/policy_network.pt", map_location=torch.device('cpu'))
controller.device = 'cpu'

obs_transformer = ObservationTransformer()

while True:
    observation = obs_transformer.transform(observation)
    observation = torch.tensor(observation).float()
    action = controller(observation).flatten().tolist()
    [observation, reward, done, info] = client.env_step(action)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()