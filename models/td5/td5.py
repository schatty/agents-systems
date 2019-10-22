import copy
import numpy as np
import time
import torch
import torch.nn.functional as F

from utils.logger import Logger
from .networks import PolicyNetwork, ValueNetwork
from .utils import _l2_project


class LearnerTD5(object):
    def __init__(self, config, learner_w_queue, log_dir):
        self.config = config
        self.log_dir = log_dir

        state_dim = config["state_dims"]
        action_dim = config["action_dims"]
        max_action = config["max_action"]
        min_action = config["min_action"]
        dense_size = config["dense_size"]
        discount = config["discount_rate"]
        tau = config["tau"]
        lr_policy = config["actor_learning_rate"]
        lr_value = config["critic_learning_rate"]
        policy_noise = config["policy_noise"]
        noise_clip = config["policy_clip"]
        policy_freq = config["policy_freq"]
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.learner_w_queue = learner_w_queue

        self.num_train_steps = config["steps_train"]

        self.actor = PolicyNetwork(config["state_dims"], config["action_dims"], config["max_action"], config["dense_size"])
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_policy)

        self.actor.to(self.device)
        self.actor_target.to(self.device)

        self.critic = ValueNetwork(state_dim, action_dim, dense_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_value)

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        log_dir = f"{log_dir}/learner"
        self.logger = Logger(log_dir)
        # training step to log
        self.log_every = [1, self.num_train_steps // 1000][self.num_train_steps > 1000]

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def run(self, training_on, batch_queue, update_step):
        time_start = time.time()
        while update_step.value < self.num_train_steps:
            if batch_queue.empty():
                continue
            batch = batch_queue.get()

            self._update_step(batch, update_step)
            update_step.value += 1
            if update_step.value % 50 == 0:
                print("Training step ", update_step.value)

            if not self.learner_w_queue.full():
                try:
                    params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
                    self.learner_w_queue.put(params)
                except:
                    pass

        training_on.value = 0

        duration_secs = time.time() - time_start
        duration_h = duration_secs // 3600
        duration_m = duration_secs % 3600 / 60
        print(f"Exit learner. Training took: {duration_h:} h {duration_m:.3f} min")

    def _update_step(self, batch, update_step):
        update_time = time.time()

        # Sample replay buffer
        state, action, next_state, reward, not_done = batch

        # ------- Update critic -------

        # Predict next actions with target policy network
        # Select action according to policy and add clipped noise
        noise = (
                torch.randn_like(action) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_action = (
                self.actor_target(next_state) + noise
        ).clamp(self.min_action, self.max_action)

        # Predict Z distribution with target value network
        target_value = self.critic_target.get_probs_q1(next_state, next_action.detach())
        target_z_atoms = self.critic.z_atoms

        # Batch of z-atoms
        target_Z_atoms = np.repeat(np.expand_dims(target_z_atoms, axis=0), self.batch_size,
                                   axis=0)  # [batch_size x n_atoms]
        # Value of terminal states is 0 by definition

        target_Z_atoms *= (not_done.cpu().int().numpy() == 1)

        # Apply bellman update to each atom (expected value)
        reward = reward.cpu().float().numpy()
        target_Z_atoms = reward + (target_Z_atoms * self.gamma)
        target_z_projected = _l2_project(torch.from_numpy(target_Z_atoms).cpu().float(),
                                         target_value.cpu().float(),
                                         torch.from_numpy(self.critic.z_atoms).cpu().float())

        # Get current Q distr
        #current_Q1, current_Q2, current_Q3, current_Q4 = self.critic.get_probs(state, action)
        current_Q1, current_Q2 = self.critic.get_probs(state, action)

        current_Q1 = current_Q1.to(self.device)
        current_Q2 = current_Q2.to(self.device)
        #current_Q3 = current_Q3.to(self.device)
        #current_Q4 = current_Q4.to(self.device)

        target_Q = torch.autograd.Variable(target_z_projected, requires_grad=False).cuda()

        value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)# + F.mse_loss(current_Q3, target_Q) + F.mse_loss(current_Q4, target_Q)

        value_loss = value_loss.mean()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # -------- Update actor -----------
        if self.total_it % self.policy_freq == 0:
            self.policy_loss = self.critic.Q1(state, self.actor(state))
            self.policy_loss = self.policy_loss * torch.tensor(self.critic.z_atoms).float().cuda()
            self.policy_loss = torch.sum(self.policy_loss, dim=1)
            self.policy_loss = -self.policy_loss.mean()

            self.actor_optimizer.zero_grad()
            self.policy_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if update_step.value % self.log_every == 0:
            step = update_step.value
            self.logger.scalar_summary("learner/value_loss", value_loss.item(), step)
            self.logger.scalar_summary("learner/policy_loss", self.policy_loss.item(), step)
            self.logger.scalar_summary("learner/learner_update_timing", time.time() - update_time, step)

        self.total_it += 1
