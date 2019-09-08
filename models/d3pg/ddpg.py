import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from .utils import OUNoise
from env.utils import create_env_wrapper
from utils.logger import Logger
from utils.reward_plot import plot_rewards
from utils.misc import empty_torch_queue


class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, num_states, num_actions, hidden_size, init_w=3e-3, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): number of neurons in hidden layers
            init_w:
        """
        super(ValueNetwork, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_actions + num_states)
        self.linear1 = nn.Linear(num_states + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, init_w=3e-3, device='cuda', action_func=None):
        """
        Args:
            num_states (int): state dimension
            num_actions (action): action dimension
            hidden_size (int): hidden size dimension
            init_w (float): margins of initialization
        """
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.action_func = action_func

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))

        if self.action_func is not None:
            x = self.action_func(x)

        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action


class LearnerD3PG(object):
    """Policy and value network update routine. """

    def __init__(self, config, log_dir=''):
        """
        Args:
            config (dict): configuration
            batch_queue (multiproc.queue): queue with batches of replays
        """
        hidden_dim = config['dense_size']
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        state_dim = config['state_dims']
        action_dim = config['action_dims']
        self.num_train_steps = config['num_steps_train']
        self.device = config['device']
        self.max_steps = config['max_ep_length']
        self.frame_idx = 0
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.log_dir = log_dir
        self.prioritized_replay = config['replay_memory_prioritized']

        # training step to log
        self.log_every = [1, self.num_train_steps // 1000][self.num_train_steps > 1000]

        log_path = f"{log_dir}/learner.pkl"
        self.logger = Logger(log_path)

        # Noise process
        env = create_env_wrapper(config)
        self.ou_noise = OUNoise(env.get_action_space())

        # Action transformation
        action_func = env.get_action_func()
        del env

        # Value and policy nets
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device, action_func=action_func)
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device, action_func=action_func)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss(reduction='none')

    def ddpg_update(self, batch, update_step, learner_w_queue, min_value=-np.inf, max_value=np.inf):
        update_time = time.time()
        state, action, reward, next_state, done, gammas, weights, inds = batch

        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().unsqueeze(1).to(self.device)
        done = torch.from_numpy(done).float().unsqueeze(1).to(self.device)

        # ------- Update critic -------

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        # Update priorities in buffer
        value_loss = value_loss.mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------- Update actor --------

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if not learner_w_queue.full():
            params = [p.data.cpu().detach().numpy() for p in self.policy_net.parameters()]
            learner_w_queue.put(params)

        # Soft update
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        if update_step.value % self.log_every == 0:
            self.logger.scalar_summary("update_step", update_step.value)
            self.logger.scalar_summary("value_loss", value_loss.item())
            self.logger.scalar_summary("policy_loss", policy_loss.item())
            self.logger.scalar_summary("learner_update_timing", time.time() - update_time)

    def run(self, training_on, batch_queue, learner_w_queue, update_step):
        while update_step.value < self.num_train_steps:
            if batch_queue.empty():
                continue
            batch = batch_queue.get()

            self.ddpg_update(batch, update_step, learner_w_queue)
            update_step.value += 1
            if update_step.value % 1 == 0:
                print("Training step ", update_step.value)

        training_on.value = 0
        empty_torch_queue(learner_w_queue)

        plot_rewards(self.log_dir)
        print("Exit learner.")