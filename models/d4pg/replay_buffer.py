'''
## Prioritised Experience Replay (PER) Memory ##
# Adapted from: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# Creates prioritised replay memory buffer to add experiences to and sample batches of experiences from
'''

import os
import pickle
import numpy as np
import random

from .segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), save_dir="~"):
        self.max_size = max_size
        self.save_dir = save_dir
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done, *args):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, *args):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
            None,
            None,
            None
        )

    def __len__(self):
        return self.size

    def dump(self):
        fn = os.path.join(self.save_dir, 'replay_buffer.pkl')
        buffer = {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done
        }
        with open(fn, 'wb') as f:
            pickle.dump(buffer, f)
        print("Buffer dumped.")


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, save_dir):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self.it_capacity = 1
        while self.it_capacity < size * 2:  # We use double the soft capacity of the PER for the segment trees to allow for any overflow over the soft capacity limit before samples are removed
            self.it_capacity *= 2

        self._it_sum = SumSegmentTree(self.it_capacity)
        self._it_min = MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"

        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def remove(self, num_samples):
        super().remove(num_samples)
        self._it_sum.remove_items(num_samples)
        self._it_min.remove_items(num_samples)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
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
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb') as f:
            pickle.dump(self._storage, f)
        print(f"Buffer dumped to {fn}")


def create_replay_buffer(config):
    size = config['replay_mem_size']
    if config['replay_memory_prioritized']:
        alpha = config['priority_alpha']
        return PrioritizedReplayBuffer(size=size, alpha=alpha)
    return ReplayBuffer(state_dim=config["state_dim"],
                        action_dim=config["action_dim"],
                        max_size=size)