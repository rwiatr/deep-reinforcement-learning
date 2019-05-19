import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.optim as optim

from drl.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hyper_space(name_values={'lr': [5e-8, 5e-1]}, n=5, d=0):
    property_matrix = np.zeros([len(name_values), n])
    for idx, key in enumerate(name_values):
        f, t = name_values[key]
        np.clip(np.linspace(f, t, n, endpoint=True), f, t, out=property_matrix[idx])

    displacement = (np.random.rand(n - 2) - .5) * 2 * d
    property_matrix[:, 1:-1] -= property_matrix[:, 1:-1] * displacement

    index_vector = np.zeros(len(name_values), dtype='int64')
    while True:
        properties = DqnAgentProperties()
        for idx, key in enumerate(name_values):
            value = property_matrix[idx, index_vector[idx]]
            if key in ['buffer_size', 'batch_size', 'update_every', 'fc_size']:
                value = int(value)
            setattr(properties, key, value)
        yield properties

        k = 0
        while True:
            index_vector[k] = (index_vector[k] + 1) % n
            if index_vector[k] == 0:
                k += 1
            else:
                break

            if k == len(name_values):
                return


class DqnAgentProperties:
    def __init__(self,
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 tau=1e-3,
                 lr=5e-4,
                 update_every=4,
                 fc_size=64):
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = int(update_every)
        self.fc_size = int(fc_size)

    def __str__(self):
        return "buffer_size={};batch_size={};gamma={};tau={};lr={};update_every={};fc_size={}" \
            .format(self.buffer_size, self.batch_size, self.gamma, self.tau, self.lr, self.update_every, self.fc_size)


class DqnAgent(Agent):
    """This class is based on a Deep Reinforcement Learning Nanodegree provided by Udacity.
       Agent interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, properties=DqnAgentProperties()):
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
        self.properties = properties

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed,
                                       fc1_units=properties.fc_size,
                                       fc2_units=properties.fc_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed,
                                        fc1_units=properties.fc_size,
                                        fc2_units=properties.fc_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=properties.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, properties.buffer_size, properties.batch_size, seed)
        # Initialize time step (for updating every properties.update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every properties.update_every time steps.
        self.t_step = (self.t_step + 1) % self.properties.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.properties.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.properties.gamma)

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

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()  # qnetwork_local learning step
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.properties.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def __str__(self):
        return "dqn_agent::{}".format(str(self.properties))

    def save(self, directory, file=None):
        if file is None:
            file = str(self)
        torch.save(self.qnetwork_target.state_dict(), directory + '/' + file + '.target')
        torch.save(self.qnetwork_local.state_dict(), directory + '/' + file + '.local')

    def load(self, directory, file=None):
        if file is None:
            file = str(self)
        target = directory + '/' + file + '.target'
        local = directory + '/' + file + '.local'
        self.qnetwork_target.load_state_dict(torch.load(target))
        self.qnetwork_local.load_state_dict(torch.load(local))
        print('done loading')


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
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
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        print(self)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
