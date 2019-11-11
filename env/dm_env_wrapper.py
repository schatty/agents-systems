from dm_control import suite
import numpy as np


class DMSuiteEnvWrapper:
    def __init__(self, env_name):
        self.env_name, self.task = env_name.split('-')
        self.env = suite.load(domain_name=self.env_name, task_name=self.task)

    def reset(self):
        time_step = self.env.reset()
        next_state = []
        for v in time_step.observation.values():
            if isinstance(v, float):
                next_state += [v]
            else:
                next_state += v.tolist()
        return np.array(next_state)

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        time_step = self.env.step(action)

        # Flatten state
        next_state = []
        for v in time_step.observation.values():
            if isinstance(v, float):
                next_state += [v]
            else:
                next_state += v.tolist()

        # Get reward
        try:
            reward = float(time_step.reward)
        except:
            reward = 0

        return np.array(next_state), reward, 0

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward