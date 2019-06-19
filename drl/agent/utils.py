import copy

import numpy as np
import random


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

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        return self

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


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
