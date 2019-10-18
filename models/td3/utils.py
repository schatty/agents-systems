import numpy as np
import pickle
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		# TODO: Get device from config
		self.device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.tensor(self.state[ind]).float().to(self.device),
			torch.tensor(self.action[ind]).float().to(self.device),
			torch.tensor(self.next_state[ind]).float().to(self.device),
			torch.tensor(self.reward[ind]).float().to(self.device),
			torch.tensor(self.not_done[ind]).float().to(self.device)
		)

	def __len__(self):
		return self.size

	def save_obs_stats(self):
		info = {
			'min': np.min(self.state, axis=0),
			'max': np.max(self.state, axis=0),
			'mean': np.mean(self.state, axis=0),
			'std': np.std(self.state, axis=0)
		}
		with open('/home/igor/replay_buffer_info.pkl', 'wb') as f:
			pickle.dump(info, f)
