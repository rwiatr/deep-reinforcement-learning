import numpy as np
import torch
import torch.nn.functional as F

from drl.agent.base import BaseAgent
from drl.agent.utils import unmap, ReplayBuffer
from drl.network.head import vanilla_local_acn, vanilla_target_acn


class Agent(BaseAgent):

    def __init__(self, conf):
        super().__init__(conf)
        self.memory = ReplayBuffer(conf.buffer_size, conf.s_dim, conf.a_dim, conf.seed)
        self.target = vanilla_target_acn(conf.s_dim, conf.a_dim).to(conf.device)
        self.local = vanilla_local_acn(conf.s_dim, conf.a_dim, conf.lr_a, conf.lr_c).to(conf.device)

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.conf.device)
        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()
        if self.conf.noise:
            action += self.conf.noise.sample()

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.conf.batch_size:
            experiences = torch.from_numpy(self.memory.sample(self.conf.batch_size)).to(self.conf.device)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = unmap(experiences, self.conf.s_dim, self.conf.a_dim)
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
