import argparse
from osim.env import L2M2019Env

from agents import TensorforcePPOAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or submit agent.")
    parser.add_argument('-t', '--train', action='store', dest='nb_steps',
            help='train agent')
    args = parser.parse_args()

    env = L2M2019Env(visualize=False)
    agent = TensorforcePPOAgent(env.observation_space, env.action_space)

    observation = env.reset()
    project_on = True
    for i in range(1):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action, project=project_on)
        if not project_on:
            print("Observation keys: ", observation.keys())
        print("Observation Size: ", len(observation))
        print("Action size: ", len(action))

    if args.nb_steps:
        # Train agent
        print("Train agent")
        agent.train(env, int(args.nb_steps))
    else:
        # Test trained agent
        print("Test agent")
        agent.test(env)
    print("Done.")
