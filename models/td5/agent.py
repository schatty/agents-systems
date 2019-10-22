import shutil
import os
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import torch

from .networks import PolicyNetwork
from env.utils import create_env_wrapper
from utils.logger import Logger
from utils.misc import make_gif, empty_torch_queue
from utils.exploration import create_epsilon_func


class Agent(object):

    def __init__(self, config, policy, global_episode, n_agent=0, log_dir='', agent_type='exploration'):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir
        self.max_action = config['max_action']
        self.min_action = config['min_action']
        self.expl_noise_mean = config['explr_noise_mean']
        self.agent_type = agent_type
        self.device = config["device"]

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.actor = policy
        #self.actor.eval()

        # Logger
        log_dir = f"{log_dir}/{agent_type}-agent-{n_agent}"
        self.logger = Logger(log_dir)

    def run(self, training_on, replay_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        while training_on.value:
            episode_reward = 0
            episode_reward_orig = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            done = False
            while not done:
                if self.agent_type == "exploration":
                    action = (self.select_action(state) + np.random.normal(self.expl_noise_mean, self.max_action * self.config['expl_noise'], size=self.config['action_dims'])
                    ).clip(self.min_action, self.max_action)
                else:
                    action = self.select_action(state)

                next_state, (reward_orig, reward), done = self.env_wrapper.step(action)

                episode_reward += reward
                episode_reward_orig += reward_orig
                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate']
                    try:
                        discounted_reward = self.env_wrapper.normalize_reward(discounted_reward)
                        replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done])
                    except:
                        pass

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                        try:
                            discounted_reward = self.env_wrapper.normalize_reward(discounted_reward)
                            replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done])
                        except:
                            pass
                    break

                num_steps += 1

            # Log metrics
            step = update_step.value
            self.logger.scalar_summary("agent/reward", episode_reward, step)
            self.logger.scalar_summary("agent/reward_orig", episode_reward_orig, step)
            self.logger.scalar_summary("agent/episode_timing", time.time() - ep_start_time, step)

            # Saving agent
            if self.local_episode % self.num_episode_save == 0 or episode_reward > best_reward:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)

        # Save replay from the first agent only
        if self.n_agent == 0:
            self.save_replay_gif()

        empty_torch_queue(replay_queue)

        self.env_wrapper.close()
        print(f"Agent {self.n_agent} done.")

    def select_action(self, state):
        state = torch.from_numpy(state.reshape(1, -1)).float().to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self):
        if self.config['env'] in ['LearnToMove']:
            return

        dir_name = "replay_render"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        for step in range(self.max_steps):
            action = self.select_action(state)
            #action = action.cpu().detach().numpy()
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")