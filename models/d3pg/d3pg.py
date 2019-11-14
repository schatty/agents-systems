import time
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import queue

from env.utils import create_env_wrapper
from utils.misc import OUNoise, empty_torch_queue
from utils.logger import Logger
from .networks import ValueNetwork


class LearnerD3PG(object):
    """Policy and value network update routine. """

    def __init__(self, config, policy, target_policy, learner_w_queue, log_dir=''):
        """
        Args:
            config (dict): configuration
        """
        self.config = config
        hidden_dim = config['dense_size']
        value_lr = config['critic_learning_rate']
        policy_lr = config['actor_learning_rate']
        state_dim = config['state_dim']
        action_dim = config['action_dim']
        self.num_train_steps = config['num_steps_train']
        self.device = config['device']
        self.max_steps = config['max_ep_length']
        self.frame_idx = 0
        self.batch_size = config['batch_size']
        self.gamma = config['discount_rate']
        self.tau = config['tau']
        self.log_dir = log_dir
        self.logger = Logger(f"{log_dir}/learner")
        self.learner_w_queue = learner_w_queue
        self.eval_freq = config['eval_freq']

        # Noise process
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])

        # Value and policy nets
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.policy_net = policy#PolicyNetwork(state_dim, action_dim, hidden_dim, device=self.device)
        self.target_value_net = copy.deepcopy(self.value_net)
        self.target_policy_net = target_policy#copy.deepcopy(self.policy_net)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss(reduction='none')

    def _update_step(self, batch, update_step, min_value=-np.inf, max_value=np.inf):
        update_time = time.time()

        state, action, reward, next_state, not_done = batch

        # Move to CUDA
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        not_done = torch.from_numpy(not_done).float().to(self.device)

        # ------- Update critic -------

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + not_done * self.gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
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

        # Soft update
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        # Send updated learner to the queue
        if self.config["agent_device"] == "cpu":# and not self.learner_w_queue.full():
            try:
                params = [p.data.to(self.config["agent_device"]).detach().numpy() for p in self.policy_net.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass

        # Logging
        step = update_step.value
        if (step+1) % self.config["eval_freq"] == 0:
            self.logger.scalar_summary("learner/update_time", time.time() - update_time, step)
            reward = self.eval_policy()
            self.logger.scalar_summary("learner/eval_reward", reward, update_step.value)
            self.logger.scalar_summary("learner/policy_loss", policy_loss.item(), step)
            self.logger.scalar_summary("learner/value_loss", value_loss.item(), step)

    def run(self, training_on, batch_queue, update_step):
        time_start = time.time()
        while update_step.value < self.num_train_steps:
            try:
                batch = batch_queue.get_nowait()
            except queue.Empty:
                continue
            self._update_step(batch, update_step)

            update_step.value += 1
            if update_step.value % 1000 == 0:
                print("Training step ", update_step.value)

        training_on.value = 0
        empty_torch_queue(self.learner_w_queue)
        print("Exit learner.")
        time_elapsed = time.time() - time_start
        hh = time_elapsed // 3600
        mm = (time_elapsed % 3600) / 60
        print(f"Training took {hh} hours {mm:.3} minutes")

    def eval_policy(self, eval_episodes=10):
        env_wrapper = create_env_wrapper(self.config)
        exp_buffer = deque()
        avg_reward = 0
        for _ in range(eval_episodes):
            state = env_wrapper.reset()
            done = False
            while not done:
                action = self.target_policy_net.get_action(state).detach().cpu().numpy().flatten()
                next_state, reward, done = env_wrapper.step(action)

                avg_reward += reward

                state = env_wrapper.normalise_state(state)
                reward = env_wrapper.normalise_reward(reward)
                exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']
                    for (_, _, r_i) in exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate']

                state = next_state

                if done:
                    # add rest of experiences remaining in buffer
                    while len(exp_buffer) != 0:
                        state_0, action_0, reward_0 = exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                    break

        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward