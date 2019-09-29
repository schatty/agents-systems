import copy
import time
import torch
import torch.nn.functional as F

from utils.logger import Logger
from utils.misc import empty_torch_queue
from .networks import PolicyNetwork, ValueNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearnerTD3(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir

        state_dim = config["state_dims"]
        action_dim = config["action_dims"]
        max_action = config["max_action"]
        dense_size = config["dense_size"]
        discount = config["discount_rate"]
        tau = config["tau"]
        lr_policy = config["actor_learning_rate"]
        lr_value = config["critic_learning_rate"]
        policy_noise = config["policy_noise"]
        noise_clip = config["policy_clip"]
        policy_freq = config["policy_freq"]

        self.num_train_steps = config["steps_train"]

        self.actor = PolicyNetwork(state_dim, action_dim, max_action, dense_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_policy)

        self.critic = ValueNetwork(state_dim, action_dim, dense_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_value)

        self.max_action = max_action
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
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def run(self, training_on, batch_queue, learner_w_queue, update_step):
        time_start = time.time()
        while update_step.value < self.num_train_steps:
            if batch_queue.empty():
                continue
            batch = batch_queue.get()

            self._update_step(batch, update_step, learner_w_queue)
            update_step.value += 1
            if update_step.value % 50 == 0:
                print("Training step ", update_step.value)

        training_on.value = 0
        empty_torch_queue(learner_w_queue)

        duration_secs = time.time() - time_start
        duration_h = duration_secs // 3600
        duration_m = duration_secs % 3600 / 60
        print(f"Exit learner. Training took: {duration_h:} h {duration_m:.3f} min")

    def _update_step(self, batch, update_step, learner_w_queue):
        update_time = time.time()

        # Sample replay buffer
        state, action, next_state, reward, not_done = batch

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            policy_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Send updated learner to the queue
        if not learner_w_queue.full():
            params = [p.data.cpu().detach().numpy() for p in self.actor.parameters()]
            learner_w_queue.put(params)

        if update_step.value % self.log_every == 0:
            step = update_step.value
            self.logger.scalar_summary("learner/value_loss", value_loss.item(), step)
            self.logger.scalar_summary("learner/policy_loss", policy_loss.item(), step)
            self.logger.scalar_summary("learner/learner_update_timing", time.time() - update_time, step)

        self.total_it += 1
