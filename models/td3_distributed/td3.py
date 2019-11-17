import copy
import time
import torch
import torch.nn.functional as F

from utils.logger import Logger
from utils.misc import empty_torch_queue
from .networks import PolicyNetwork, ValueNetwork
from env.utils import create_env_wrapper


class LearnerTD3(object):
    def __init__(self, config, local_policy, target_policy, learner_w_queue, log_dir):
        self.config = config
        self.log_dir = log_dir

        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        max_action = config["action_high"]
        dense_size = config["dense_size"]
        discount = config["discount_rate"]
        tau = config["tau"]
        lr_policy = config["actor_learning_rate"]
        lr_value = config["critic_learning_rate"]
        policy_noise = config["policy_noise"]
        noise_clip = config["noise_clip"]
        policy_freq = config["policy_freq"]

        self.num_train_steps = config["num_steps_train"]

        self.actor = local_policy
        self.target_policy_net = target_policy
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_policy)

        self.critic = ValueNetwork(state_dim, action_dim, dense_size).to(config["device"])
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_value)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.learner_w_queue = learner_w_queue

        self.total_it = 0

        log_dir = f"{log_dir}/learner"
        self.logger = Logger(log_dir)
        # training step to log
        self.log_every = [1, self.num_train_steps // 1000][self.num_train_steps > 1000]

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.config["device"])
        return self.actor(state).cpu().data.numpy().flatten()

    def run(self, training_on, batch_queue, update_step):
        time_start = time.time()
        while update_step.value < self.num_train_steps:
            if batch_queue.empty():
                continue
            batch = batch_queue.get()

            self._update_step(batch, update_step)
            update_step.value += 1
            if update_step.value % 1000 == 0:
                print("Training step ", update_step.value)

        training_on.value = 0

        duration_secs = time.time() - time_start
        duration_h = duration_secs // 3600
        duration_m = duration_secs % 3600 / 60
        print(f"Exit learner. Training took: {duration_h:} h {duration_m:.3f} min")

    def _update_step(self, batch, update_step):
        update_time = time.time()

        # Sample replay buffer
        state, action, next_state, reward, not_done = batch

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.target_policy_net(next_state) + noise
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
            self.policy_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.policy_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Send updated learner to the queue
        if self.config["agent_device"] == "cpu":
            try:
                params = [p.data.to(self.config["agent_device"]).detach().numpy() for p in self.policy_net.parameters()]
                self.learner_w_queue.put_nowait(params)
            except:
                pass

        step = update_step.value
        if (step + 1) % self.config["eval_freq"] == 0:
            self.logger.scalar_summary("learner/update_time", time.time() - update_time, step)
            reward = self.eval_policy()
            self.logger.scalar_summary("learner/eval_reward", reward, update_step.value)
            self.logger.scalar_summary("learner/policy_loss", self.policy_loss.item(), step)
            self.logger.scalar_summary("learner/value_loss", value_loss.item(), step)

        self.total_it += 1

    def eval_policy(self, eval_episodes=10):
        env_wrapper = create_env_wrapper(self.config)
        avg_reward = 0
        for _ in range(eval_episodes):
            state = env_wrapper.reset()
            done = False
            while not done:
                action = self.target_policy_net.get_action(state).detach().cpu().numpy().flatten()
                next_state, reward, done = env_wrapper.step(action)
                avg_reward += reward
                state = next_state
                if done:
                    break

        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward
