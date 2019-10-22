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
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    z_p = torch.tensor(z_p).float()

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]

    d_pos = torch.cat([z_q, vmin[None]], 0)[1:]
    d_neg = torch.cat([vmax[None], z_q], 0)[:-1]

    # Clip z_p to be in new support range (vmin, vmax)
    z_p = torch.clamp(z_p, vmin, vmax)[:, None, :]

    # Get the distance between atom values in support
    d_pos = (d_pos - z_q)[None, :, None]
    d_neg = (z_q - d_neg)[None, :, None]
    z_q = z_q[None, :, None]

    d_neg = torch.where(d_neg>0, 1./d_neg, torch.zeros(d_neg.shape))
    d_pos = torch.where(d_pos>0, 1./d_pos, torch.zeros(d_pos.shape))

    delta_qp = z_p - z_q
    d_sign = (delta_qp >= 0).type(p.dtype)

    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]
    return torch.sum(torch.clamp(1. - delta_hat, 0., 1.) * p, -1)