import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from drl.agent.original.model import Actor, Critic

# BUFFER_SIZE = int(1e5)  # replay buffer size
from drl.network.body import ActionFCNet, vanilla_action_fc_net, FCNet
from drl.network.head import ActorCriticNet, vanilla_acn

BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, conf):
        self.conf = conf
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        if conf.a:
            actor_local = Actor(state_size, action_size, random_seed).to(conf.device)
            actor_target = Actor(state_size, action_size, random_seed).to(conf.device)
        else:
            actor_local = FCNet(l_dims=(conf.s_dim, 400, 300, conf.a_dim), actv=(F.relu, F.relu, F.tanh)).to(
                conf.device)
            actor_target = FCNet(l_dims=(conf.s_dim, 400, 300, conf.a_dim), actv=(F.relu, F.relu, F.tanh)).to(
                conf.device)
        actor_optimizer = optim.Adam(actor_local.parameters(), lr=conf.lr_a)

        # Critic Network (w/ Target Network)
        # self.critic_local = vanilla_action_fc_net(conf.s_dim, conf.a_dim).to(conf.device)
        if conf.a:
            critic_local = Critic(state_size, action_size, random_seed).to(conf.device)
            critic_target = Critic(state_size, action_size, random_seed).to(conf.device)
        else:
            critic_local = vanilla_action_fc_net(conf.s_dim, conf.a_dim).to(conf.device)
            critic_target = vanilla_action_fc_net(conf.s_dim, conf.a_dim).to(conf.device)
        critic_optimizer = optim.Adam(critic_local.parameters(), lr=conf.lr_c, weight_decay=WEIGHT_DECAY)

        self.local = ActorCriticNet(actor_local, critic_local, actor_optimizer, critic_optimizer)
        self.target = vanilla_acn(conf.s_dim, conf.a_dim)

        # Noise process
        self.noise = conf.noise

        # Replay memory
        self.memory = ReplayBuffer(action_size, conf.buffer_size, conf.batch_size, random_seed, conf)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.conf.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.conf.device)
        # self.actor_local.eval()
        self.local.eval()

        with torch.no_grad():
            # action = self.actor_local(state).cpu().data.numpy()
            action = self.local.actor(state).cpu().data.numpy()

        # self.actor_local.train()
        self.local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences  # unmap(experiences, self.conf.s_dim, self.conf.a_dim)

        # ### UPDATE CRITIC LOCAL ### #
        next_actions = self.target.actor(next_states)  # u'(s_t+1) = ~a_t+2
        # calculate the prediction for the next_Q_targets (cumulative reward from next step)
        next_q_targets = self.target.critic(next_states, next_actions)  # Q'(s_t+1, ~a_t+2)
        # cumulative reward from this step is reward + discounted cumulative reward from next step (next_Q_targets)
        q_targets = rewards + (self.conf.gamma * next_q_targets * (1 - dones))

        # Q(s_t, a_t)
        expected_q_targets = self.local.critic(states, actions)
        # Optimizing local critic using Q(s_t, a_t) - R + gamma * Q'(s_t+1, u'(s_t+1)) by minimising the loss
        self.local.learn_critic(F.mse_loss(expected_q_targets, q_targets))

        # ### UPDATE ACTOR LOCAL ### #
        # a_t+1 <- u(s_t)
        actions = self.local.actor(states)
        # -mean(Q(s_t, a_t+1))
        self.local.learn_actor(-self.local.critic(states, actions).mean())

        self.target.update(self.local, self.conf.tau)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, conf):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.conf = conf
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(
            self.conf.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(
            self.conf.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(
            self.conf.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.conf.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.conf.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
