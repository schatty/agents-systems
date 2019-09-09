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
        target_v_field = np.array([observation['v_tgt_field'][0, 5, 5], observation['v_tgt_field'][1, 5, 5]])
        features += target_v_field.tolist()
        features = np.asarray(features)

        return features


class LearnToMove(EnvWrapper):
    def __init__(self, config):
        super(LearnToMove, self).__init__(config)

        self.env = L2M2019Env(visualize=bool(config['visualize']), integrator_accuracy=0.001)
        self.project = True # False - dict of size 14, True - dict of size 4
        self.env.reset(project=self.project)
        self.observation_transformer = ObservationTransformer()

    def step(self, action):
        obs, reward_orig, done, _ = self.env.step(action.flatten(), project=self.project)
        done = self.transform_done(done, obs)
        obs = self.observation_transformer.transform(obs)
        obs = self.normalize_state(obs)

        return obs, (reward_orig, reward_orig), done

    def get_action_space(self):
        class ActionSpace(object):
            def __init__(self):
                self.low = 0
                self.high = 1
                self.shape = (22,)
        return ActionSpace()

    def transform_done(self, done, obs):
        return done

    def reset(self):
        obs = self.observation_transformer.transform(self.env.reset(project=self.project))
        obs = self.normalize_state(obs)
        return obs
