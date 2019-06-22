import torch
from wheel.cli import unpack

import drl.agent.DDPG as ddpg
from drl.agent.base import BaseAgent
from drl.agent.original import ddpg_agent
from drl.agent.utils import ReplayBuffer, unmap


def multi_ddpg_with_shared_mem(conf, n):
    conf.mem_disabled = True
    conf.seed = None
    return Agent(conf, list([ddpg.Agent(conf) for _ in range(n)]))


class Agent(BaseAgent):
    def __init__(self, conf, agents, update_every_n_steps=None):
        super().__init__(conf)
        self.memory = ReplayBuffer(conf.buffer_size, conf.s_dim, conf.a_dim, conf.seed)
        self.agents = agents
        self.update_every_n_steps = update_every_n_steps
        self.step_n = 0

    def act(self, states):
        return list(a.act(s) for a, s in zip(self.agents, states))

    def step(self, states, actions, rewards, next_states, dones):
        self.step_n += 1
        for idx in range(len(self.agents)):
            self.memory.add(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])

        if not self.update_every_n_steps:
            self.try_update()
        elif self.step_n > self.update_every_n_steps:
            self.try_update()
            self.step_n = 0

    def try_update(self):
        if len(self.memory) > self.conf.batch_size:
            for agent in self.agents:
                sample = self.memory.sample(self.conf.batch_size)
                sample = torch.from_numpy(sample).to(self.conf.device)
                if isinstance(agent, ddpg_agent.Agent):
                    sample = unmap(sample, self.conf.s_dim, self.conf.a_dim)
                agent.learn(sample)

    def learn(self, experiences):
        """ NoOp """
