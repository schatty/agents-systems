import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from model import ActorCriticNetwork


class Agent:

    def __init__(self, environment, brain_name, num_agents,
                 state_size, action_size):
        self.environment = environment
        self.brain_name = brain_name
        self.num_agents = num_agents
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), 2e-4, eps=1e-5)

        self.discount_rate = 0.99
        self.tau = 0.95
        self.learning_rounds = 10
        self.ppo_clip = 0.2
        self.gradient_clip = 5
        self.mini_batch_number = 64

    def generate_rollout(self):
        rollout = []
        episode_rewards = np.zeros(self.num_agents)

        # Reset environment
        env_info = self.environment.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations

        # Generate rollout from an entire episode
        while True:
            states = torch.Tensor(states).cuda()
            actions, log_probs, values = self.network(states)
            actions_np = actions.cpu().detach().numpy()
            env_info = self.environment.step(actions_np)[self.brain_name]
            rewards = env_info.rewards
            dones = np.array(env_info.local_done)
            next_states = env_info.vector_observations

            episode_rewards += rewards
            rollout.append([states, values.detach(),
                            actions.detach(), log_probs.detach(),
                            rewards, 1-dones])
            states = next_states

            if np.any(dones):
                break

        states = torch.Tensor(states).cuda()
        _, _, last_value = self.network(states)
        rollout.append([states, last_value, None, None, None, None])
        return rollout, last_value, episode_rewards

    def process_rollout(self, rollout, last_value):
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.zeros((self.num_agents, 1)).cuda()
        returns = last_value.detach()

        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, dones = rollout[i]
            dones = torch.Tensor(dones).unsqueeze(1).cuda()
            rewards = torch.Tensor(rewards).unsqueeze(1).cuda()
            next_value = rollout[i + 1][1]
            returns = rewards + self.discount_rate * dones * returns

            td_error = rewards + self.discount_rate * dones * next_value.detach() - value.detach()
            advantages = advantages * self.tau * self.discount_rate * dones + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        return processed_rollout

    def train_network(self, states, actions, log_probs_old, 
                      returns, advantages):
        batcher = Batcher(states.size(0) // self.mini_batch_number, [np.arange(states.size(0))])
        for _ in range(self.learning_rounds):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages 
                obj_clipped = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * sampled_advantages 
                policy_loss = -torch.min(obj, obj_clipped).mean(0)

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                self.optimizer.step()

    def step(self):
        # Run a single episode to generate a rollout 
        rollout, last_value, episode_rewards = self.generate_rollout()
        # Process the rollout to calculate advantages 
        processed_rollout = self.process_rollout(rollout, last_value) 
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        # Train the network
        self.train_network(states, actions, log_probs_old, returns, advantages)
        # Return the average reward across all agents for this episode
        return np.mean(episode_rewards)


class Batcher:

    def __init__(self, batch_size, data):
        self.batch_size = batch_size 
        self.data = data 
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0 
        self.batch_end = self.batch_start + self.batch_size 

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end 
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch 

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
        self.reset()
