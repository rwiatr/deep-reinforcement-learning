import numpy as np
import torch
import torch.nn.functional as F

from drl.agent.utils import unmap, ReplayBuffer2, OUNoise
from drl.env import BaseAgent
from drl.network.head import default_acn


class Agent(BaseAgent):

    def __init__(self, conf, device):
        super().__init__(conf)
        self.device = device
        self.memory = ReplayBuffer2(conf.buffer_size, conf.seed, device)
        self.target = default_acn(conf.s_dim, conf.a_dim).to(device)
        self.local = default_acn(conf.s_dim, conf.a_dim, conf.lr_a, conf.lr_c, wd_a=conf.wd_a, wd_c=conf.wd_c) \
            .to(device)
        self.noise = OUNoise(conf.a_dim, conf.seed)

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()

        action += self.noise.sample()

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.conf.batch_size:
            sample = self.memory.sample(self.conf.batch_size)
            # sample = torch.from_numpy(sample).to(self.device)
            # sample = self.memory.sample()
            self.learn(sample)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

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

    def __str__(self):
        return "ddpg={" + str(self.conf) + "}"

    def save(self, path, name):
        torch.save(self.target.state_dict(), path + '/' + name + '.target')
        torch.save(self.local.state_dict(), path + '/' + name + '.local')
