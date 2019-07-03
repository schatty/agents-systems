import argparse

from osim.env import ProstheticsEnv
from osim.http.client import Client

from helper.wrappers import DictToListFull, ForceDictObservation, JSONable
from helper.baselines import * 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI for Prosthetics")
    parser.add_argument('agent', help='agent class name')
    parser.add_argument('-t', '--train', action='store', dest='nb_steps', help='Train agent')
    parser.add_argument('-v', '--visualize', action='store_true', default=False, help='render the environment')
    args = parser.parse_args()

    if args.agent not in globals():
        raise ValueError(f'[run] Agent {args.agent} not found.')
    SpecifiedAgent = globals()[args.agent]

    if args.nb_steps:
        # Train agent localy
        env = ProstheticsEnv(visualize=args.visualize)
        env = ForceDictObservation(env)
        env = DictToListFull(env)
        env = JSONable(env)
        agent = SpecifiedAgent(env.observation_space, env.action_space)
        agent.train(env, int(args.nb_steps))
    else:
        # Test agent locally
        env = ProstheticsEnv(visualize=args.visualize)
        env = ForceDictObservation(env)
        env = DictToListFull(env)
        env = JSONable(env)
        agent = SpecifiedAgent(env.observation_space, env.action_space)
        agent.test(env)
