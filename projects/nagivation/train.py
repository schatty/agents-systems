import os
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

from unityagents import UnityEnvironment
import numpy as np


def evaluate(env, agent, n_episodes=10):
    """Run evaluation of an agent. 
    
    Args:
        env: environment.
        agent: agent.
        n_episode: number of evaluation to take the mean of.
    """
    env.eval_mode()
    ep_scores = []
    for _ in range(n_episodes):
        state = env.reset()
        score = 0                                      
        while True:
            action = agent.select_action(state)            
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state
            if done:                                    
                break
        ep_scores.append(score)

    print("Scores: ", ep_scores)
    return np.mean(ep_scores)


class RandomAgent:
    """Agent that takes random actions. """

    def __init__(self, action_size):
        self.action_size = action_size
    
    def select_action(self, state):
        return np.random.randint(self.action_size)
    
    
class UnityEnvWrapper:
    def __init__(self, unity_env):
        self.env = unity_env
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.train_mode = True
        self.info = None
        
    def reset(self):
        """Return state."""
        self.info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self.info.vector_observations[0]
        
    def step(self, action):
        """Return (next_state, reward, done) tuple. """
        self.info = self.env.step(action)[self.brain_name]
        next_state = self.info.vector_observations[0]
        reward = self.info.rewards[0]
        done = self.info.local_done[0]
        return next_state, reward, done
    
    def eval_mode(self):
        self.train_mode = False
        
    def train_mode(self):
        self.train_mode = True
        
    def close(self):
        self.env.close()
        
    @property
    def action_dim(self):
        return self.brain.vector_action_space_size
    
    @property
    def state_dim(self):
        if self.info is None:
            print("Reset environment to get access to the state size.")
            return
        return len(self.info.vector_observations[0])

env = UnityEnvWrapper(UnityEnvironment(file_name="/home/igor/Banana_Linux/Banana"))
env.reset()


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=64):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Hidden size
        """
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        o = self.relu(self.linear1(state))
        o = self.relu(self.linear2(o))
        o = self.linear3(o)
        return o


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        next_states = next_states.cuda()
        dones = dones.cuda()
                
        q_values = self.qnetwork_local(states)
        q_values_next = self.qnetwork_target(next_states)
                
        q_value = q_values.gather(1, actions) # [64, 1]
        q_value_next = q_values_next.max(1)[0]
        q_value_next = q_value_next.unsqueeze(1) # [64, 1]
                
        expected_q_value = rewards + gamma * q_value_next * (1 - dones)
                
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, save_dir):
        """Save model to the given directory. 
        Args:
            save_dir (str): directory to save the model.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.qnetwork_local, f"{save_dir}/qnet_local.pt")
        torch.save(self.qnetwork_target, f"{save_dir}/qnet_target.pt")

    def load(self, load_dir):
        self.qnetwork_local = torch.load(f"{load_dir}/qnet_local.pt")
        self.qnetwork_target = torch.load(f"{load_dir}/qnet_target.pt")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


agent = Agent(state_size=env.state_dim, action_size=env.action_dim, seed=0)


def dqn(n_episodes=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.save("saved_models/model")
    return scores


if __name__ == "__main__":
    scores = dqn()
