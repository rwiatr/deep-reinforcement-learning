import numpy as np
import torch
import torch.nn.functional as F

from drl.agent.base import BaseAgent
# from drl.agent.utils import unmap, ReplayBuffer
from drl.dql.dqn_agent import ReplayBuffer
from drl.network.head import vanilla_acn


class Agent(BaseAgent):

    def __init__(self, conf):
        super().__init__(conf)
        # self.memory = ReplayBuffer(conf.buffer_size, conf.s_dim, conf.a_dim, conf.seed)
        self.memory = ReplayBuffer(conf.a_dim, conf.buffer_size, conf.batch_size, conf.seed)
        self.target = vanilla_acn(conf.s_dim, conf.a_dim).to(conf.device)
        self.local = vanilla_acn(conf.s_dim, conf.a_dim, conf.lr_a, conf.lr_c).to(conf.device)

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
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.conf.batch_size:
            # sample = self.memory.sample(self.conf.batch_size)
            # sample = torch.from_numpy(sample).to(self.conf.device)
            sample = self.memory.sample()
            self.learn(sample)

    def learn(self, experiences):
        # states, actions, rewards, next_states, dones = experiences  # unmap(experiences, self.conf.s_dim, self.conf.a_dim)
        #
        # # ### UPDATE CRITIC LOCAL ### #
        # next_actions = self.target.actor(next_states)  # u'(s_t+1) = ~a_t+2
        # # calculate the prediction for the next_Q_targets (cumulative reward from next step)
        # next_q_targets = self.target.critic(next_states, next_actions)  # Q'(s_t+1, ~a_t+2)
        # # cumulative reward from this step is reward + discounted cumulative reward from next step (next_Q_targets)
        # q_targets = rewards + (self.conf.gamma * next_q_targets * (1 - dones))
        #
        # # Q(s_t, a_t)
        # expected_q_targets = self.local.critic(states, actions)
        # # Optimizing local critic using Q(s_t, a_t) - R + gamma * Q'(s_t+1, u'(s_t+1)) by minimising the loss
        # self.local.learn_critic(F.mse_loss(expected_q_targets, q_targets))
        #
        # # ### UPDATE ACTOR LOCAL ### #
        # # a_t+1 <- u(s_t)
        # actions = self.local.actor(states)
        # # -mean(Q(s_t, a_t+1))
        # self.local.learn_actor(-self.local.critic(states, actions).mean())
        #
        # self.target.update(self.local, self.conf.tau)

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_states)
        actions_next = self.target.actor(next_states)
        # Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets_next = self.target.critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.conf.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        # Q_expected = self.critic_local(states, actions)
        Q_expected = self.local.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        self.local.learn_critic(critic_loss)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # actions_pred = self.actor_local(states)
        actions_pred = self.local.actor(states)
        # actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss = -self.local.critic(states, actions_pred).mean()
        # Minimize the loss
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        self.local.learn_actor(actor_loss)

        # ----------------------- update target networks ----------------------- #
        # self.soft_update(self.critic_local, self.critic_target, TAU)
        # self.soft_update(self.actor_local, self.actor_target, TAU)
        # self.soft_update(self.local, self.target, TAU)
        self.target.update(self.local, self.conf.tau)
