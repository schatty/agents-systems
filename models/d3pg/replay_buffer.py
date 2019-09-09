import random
import numpy as np
import pandas as pd


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create replay buffer.
        Args:
            size (int): max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = (obs_t, action, reward, obs_tp1, done, gamma)

        self._storage.append(data)

        self._next_idx += 1

    def remove(self, num_samples):
        del self._storage[:num_samples]
        self._next_idx = len(self._storage)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, gammas = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, gamma = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            gammas.append(gamma)
        return [np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            gammas)]

    def sample(self, batch_size, **kwags):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        weights = np.zeros(len(idxes))
        inds = np.zeros(len(idxes))
        return self._encode_sample(idxes) + [weights, inds]

    def upload_stats(self, dir):
        print("Start analysing buffer...")
        obs = []
        rewards = []
        for replay in self._storage:
            obs.append(replay[0])
            rewards.append(replay[2])

        stats_obs = {}
        obs = np.asarray(obs)
        stats_obs['obs_mean'] = np.mean(obs, axis=1)
        stats_obs['obs_std'] = np.std(obs, axis=1)
        stats_obs['obs_min'] = np.min(obs, axis=1)
        stats_obs['obs_max'] = np.max(obs, axis=1)
        pd.DataFrame.from_dict(stats_obs).to_csv(f"{dir}/stats_obs.csv", index=False)

        stats_reward = {}
        stats_reward['reward_mean'] = [np.mean(rewards)]
        stats_reward['reward_std'] = [np.std(rewards)]
        stats_reward['reward_min'] = [np.min(rewards)]
        stats_reward['reward_max'] = [np.max(rewards)]
        pd.DataFrame.from_dict(stats_reward).to_csv(f"{dir}/stats_reward.csv", index=False)