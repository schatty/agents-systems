import numpy as np
import pickle
import random
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


def calc_ang_to_target(rot_y, target_v):
    pv = np.array([np.cos(rot_y), -np.sin(rot_y)])
    cos_ang = np.dot(target_v, pv)
    ang = np.arccos(cos_ang)
    return ang


class RewardTransformer(object):
    def __init__(self):
        self.reward_data = []

    def transform(self, reward_orig, obs_train):
        reward = reward_orig + self._extra_reward(obs_train)

        return reward

    def _extra_reward(self, state_desc):
        reward_extra = 0

        state_desc["target_vel"] = [state_desc['v_tgt_field'][0, 5, 5], 0, state_desc['v_tgt_field'][1, 5, 5]]

        scale = 50.0

        # penalty for not bending knees
        for joint in ["knee_l", "knee_r"]:
            # make this a penalty instead of a reward: range approx [-4, 0.3] * X
            # negative is bending correct way
            penalty = (-1.0 * state_desc["joint_pos"][joint][0] - 0.5) * 0.5
            if penalty > 0.0:
                penalty = 0.0
            reward_extra += penalty

        # reduce points if the femur is far away from foot in z-axis
        # depend on movement direction
        # target vector
        tv = np.array([3.0, 0.0])
        difficulty = 0
        if "target_vel" in state_desc:
            difficulty = 1
            tv[0] = state_desc["target_vel"][0]
            tv[1] = state_desc["target_vel"][2]
        # normalize
        tv_len = np.linalg.norm(tv)
        if tv_len > 0.0:
            tv = tv / tv_len
        else:
            tv = np.array(1.0, 0.0)

        side = np.array([tv[1] * -1.0, tv[0]])  # rotate 90 CCW
        # print("side:", side)

        # keep left foot under upper leg
        femur_l_pos = np.array([state_desc["body_pos"]["femur_l"][i] for i in (0, 2)])
        toes_l_pos = np.array([state_desc["body_pos"]["toes_l"][i] for i in (0, 2)])
        diff_l_v = femur_l_pos - toes_l_pos
        diff_l = np.dot(side, diff_l_v)
        reward_extra -= (diff_l ** 2) * 5.0 / scale # max is around -3.0 with * 10.0
        # print("left diff:", diff_l)

        # keep right foot under upper leg
        femur_r_pos = np.array([state_desc["body_pos"]["femur_r"][i] for i in (0, 2)])
        foot_r_pos = np.array([state_desc["body_pos"]["toes_r"][i] for i in (0, 2)])
        diff_r_v = femur_r_pos - foot_r_pos
        diff_r = np.dot(side, diff_r_v)
        reward_extra -= (diff_r ** 2) * 5.0 / scale # max is around -4.0 with * 10.0
        # print("right diff:", diff_r)

        # orient pelvis forward => avoid weird sideways running
        pelvis_rot_y = state_desc["body_pos_rot"]["pelvis"][1]
        ang = calc_ang_to_target(pelvis_rot_y, tv)
        pelvis_dir_penalty = (ang ** 2) * 20.0 / scale
        # print(ang, pelvis_dir_penalty)
        reward_extra -= pelvis_dir_penalty

        # feet direction in target direction => encourage actual turning instead of sideways running
        if difficulty > 0:
            toes_l_rot_y = state_desc["body_pos_rot"]["toes_l"][1]
            tl_ang = calc_ang_to_target(toes_l_rot_y, tv)
            tl_penalty = (tl_ang ** 2) * 20.0 / scale
            reward_extra -= tl_penalty

            foot_r_rot_y = state_desc["body_pos_rot"]["toes_r"][1]
            fr_ang = calc_ang_to_target(foot_r_rot_y, tv)
            fr_penalty = (fr_ang ** 2) * 20.0 / scale
            reward_extra -= fr_penalty
            # print("tv:", tv, "feet dir penalty (left):", tl_ang, tl_penalty, "(right):", fr_ang, fr_penalty)

        # make feets be on correct side of each other
        # get distance projected on side vector
        feet_v = foot_r_pos - toes_l_pos
        diff_feet = np.dot(side, feet_v)
        # print("diff feet:", diff_feet)
        if diff_feet < 0.176:
            # crossing feet, this is extra bad
            reward_extra -= (diff_feet - 0.176) ** 2 * 100.0 / scale
        else:
            reward_extra -= (diff_feet - 0.176) ** 2 * 10.0 / scale

        # cap penalty, trying to keep final reward in a range of [-11,10]
        max_penalty = -5

        if reward_extra < max_penalty:
            reward_extra = max_penalty
        if reward_extra > 0.0:
            reward_extra = 0.0

        # print("reward_extra", reward_extra)
        return reward_extra


class LearnToMove(EnvWrapper):
    def __init__(self, config):
        super(LearnToMove, self).__init__(config)

        self.env = L2M2019Env(visualize=bool(config['visualize']), integrator_accuracy=0.001)
        self.project = True # False - dict of size 14, True - dict of size 4
        self.env.reset(project=self.project)

        self.observation_transformer = ObservationTransformer()
        self.reward_transformer = RewardTransformer()

    def step(self, action):
        obs_eval, reward_orig, done1, _ = self.env.step(action.flatten(), project=True)
        obs_train, _, done2, _ = self.env.step(action.flatten(), project=False)
        done = max(done1, done2)

        #self.reward_transformer.reward_data.append(reward_orig)

        reward_shaped = self.reward_transformer.transform(reward_orig, obs_train)

        done = self.transform_done(done, obs_eval)
        obs = self.observation_transformer.transform(obs_eval)
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

    def close(self):
        pass
