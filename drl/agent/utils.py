import copy
from collections import deque, namedtuple

import numpy as np
import random

import torch


def unmap(experiences, s_dim, a_dim):
    left = 0
    states = experiences[:, left:left + s_dim]
    left += s_dim
    next_states = experiences[:, left:left + s_dim]
    left += s_dim
    actions = experiences[:, left:left + a_dim]
    left += a_dim
    rewards = experiences[:, left:left + 1]
    left += 1
    dones = experiences[:, left:left + 1]

    return states, actions, rewards, next_states, dones


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.state = None
        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        return self

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer3:

    def __init__(self, buffer_size, seed, device):
        # self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBuffer2:
    def __init__(self, buffer_size, seed, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        device = self.device
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return states.to(device), actions.to(device), rewards.to(device), next_states.to(device), dones.to(device)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, s_dim, a_dim, seed=None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.memory = np.zeros([buffer_size, 2 * s_dim + a_dim + 2]).astype(np.float32)
        self.memory_pos = 0
        if seed:
            np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        m_idx = self.memory_pos % len(self.memory)
        left = 0
        self.memory[m_idx, left:left + self.s_dim] = state
        left += self.s_dim
        self.memory[m_idx, left:left + self.s_dim] = next_state
        left += self.s_dim
        self.memory[m_idx, left:left + self.a_dim] = action
        left += self.a_dim
        self.memory[m_idx, left:left + 1] = reward
        left += 1
        self.memory[m_idx, left:left + 1] = done

        self.memory_pos += 1

    def sample(self, n):
        return self.memory[np.random.choice(len(self), n, replace=False), :]

    def __len__(self):
        return min(len(self.memory), self.memory_pos)
