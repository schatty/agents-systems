import numpy as np
from osim.env import L2M2019Env

from .env_wrapper import EnvWrapper

class ObservationTransformer(object):
    """Transforms observation signal. """

    def __init__(self):
        pass

    def transform(self, observation):
        features = []

        features += [observation['pelvis']['height']]
        features += [observation['pelvis']['pitch']]
        features += [observation['pelvis']['roll']]
        features += observation['pelvis']['vel']

        for leg in ['l_leg', 'r_leg']:
            features += observation[leg]['ground_reaction_forces']
            features += [observation[leg]['joint']['hip_abd'],
                         observation[leg]['joint']['hip'],
                         observation[leg]['joint']['knee'],
                         observation[leg]['joint']['ankle']]
            features += [observation[leg]['d_joint']['hip_abd'],
                         observation[leg]['d_joint']['hip'],
                         observation[leg]['d_joint']['knee'],
                         observation[leg]['d_joint']['ankle']]
            features += [observation[leg]['HAB'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['HAD'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['HFL'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['GLU'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['HAM'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['RF'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['VAS'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['BFSH'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['GAS'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['SOL'][k] for k in ['f', 'l', 'v']]
            features += [observation[leg]['TA'][k] for k in ['f', 'l', 'v']]

        #target_v_field = observation['v_tgt_field'].flatten() # [2 x 11 x 11]

        # Fixed number
        if isinstance(observation['v_tgt_field'], np.ndarray):
            target_v_field = np.array([observation['v_tgt_field'][0, 5, 5], observation['v_tgt_field'][1, 5, 5]])
            features += target_v_field.tolist()
        else:
            features += [observation['v_tgt_field'][0][5][5], observation['v_tgt_field'][1][5][5]]
        features = np.asarray(features)

        return features


class RewardTransformer(object):
    def __init__(self):
        pass

    def transform(self, reward_orig, obs):
        r_leg = np.clip(np.abs(obs['r_leg']['joint']['hip']), 0, 0.5) / 0.5
        l_leg = np.clip(np.abs(obs['l_leg']['joint']['hip']), 0, 0.5) / 0.5

        r_leg /= 100.
        l_leg /= 100.

        reward = reward_orig + r_leg + l_leg

        return reward


class LearnToMove(EnvWrapper):
    def __init__(self, config):
        super(LearnToMove, self).__init__(config)

        self.env = L2M2019Env(visualize=bool(config['visualize']), integrator_accuracy=0.001)
        self.project = True # False - dict of size 14, True - dict of size 4
        self.env.reset(project=self.project)

        self.observation_transformer = ObservationTransformer()
        self.reward_transformer = RewardTransformer()

    def step(self, action):
        obs, reward_orig, done, _ = self.env.step(action.flatten(), project=self.project)

        reward_shaped = self.reward_transformer.transform(reward_orig, obs)

        done = self.transform_done(done, obs)
        obs = self.observation_transformer.transform(obs)
        obs = self.normalize_state(obs)

        return obs, (reward_orig, reward_shaped), done

    def get_action_space(self):
        class ActionSpace(object):
            def __init__(self):
                self.low = 0
                self.high = 1
                self.shape = (22,)
        return ActionSpace()

    def transform_done(self, done, obs):
        if obs['pelvis']['height'] < 0.7:
            return 1
        return done

    def reset(self):
        obs = self.observation_transformer.transform(self.env.reset(project=self.project))
        obs = self.normalize_state(obs)
        return obs
