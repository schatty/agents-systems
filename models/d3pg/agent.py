import shutil
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import torch

from .utils import OUNoise
from .ddpg import PolicyNetwork
from env.utils import create_env_wrapper
from utils.logger import Logger
from utils.misc import make_gif, empty_torch_queue


class Agent(object):

    def __init__(self, config, learner_w_queue, global_episode, n_agent=0, log_dir=''):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(self.env_wrapper.get_action_space())
        self.ou_noise.reset()

        self.learner_w_queue = learner_w_queue
        self.actor = PolicyNetwork(num_actions=config['action_dims'],
                                   num_states=config['state_dims'],
                                   hidden_size=config['dense_size'])
        self.actor.eval()

        # Logger
        log_path = f"{log_dir}/agent-{n_agent}.pkl"
        self.logger = Logger(log_path)

    def update_actor_learner(self):
        """Update local actor to the actor from learner. """
        if self.learner_w_queue.empty():
            return
        source = self.learner_w_queue.get()
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)

    def run(self, training_on, replay_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        while training_on.value:
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            done = False
            while not done:
                action = self.actor.get_action(state)
                action = self.ou_noise.get_action(action, num_steps)
                action = action.squeeze(0)
                next_state, reward, done = self.env_wrapper.step(action)

                episode_reward += reward

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

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
                        replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
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
                            replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                        except:
                            pass
                    break

                num_steps += 1

            # Log metrics
            self.logger.scalar_summary("update_step", update_step.value)
            self.logger.scalar_summary("reward", episode_reward)
            self.logger.scalar_summary("episode_timing", time.time() - ep_start_time)

            # Saving agent
            if self.local_episode % self.num_episode_save == 0 or episode_reward > best_reward:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)
            if self.local_episode % self.config['update_agent_ep'] == 0:
                self.update_actor_learner()

        # Save replay from the first agent only
        if self.n_agent == 0:
            self.save_replay_gif()

        empty_torch_queue(replay_queue)

        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self):
        dir_name = "replay_render"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy()
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