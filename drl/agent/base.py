import torch

from drl.agent import DDPG as ddpg
from drl.agent.original import ddpg_agent
from drl.agent.utils import ReplayBuffer, unmap
import numpy as np


class HyperSpace:
    """"""

    def __str__(self):
        return str(self.__dict__)

def hyper_space_sc_rec(map, key, keys):
    for value in map[key]:
        hyper_space_sc_rec(map, key[0], keys[1:])


def hyper_space_ns(params={'lr': [5e-8, 5e-1]}):
    all_keys = params.keys()

    for key in params.keys():
        values = params[key]


    property_matrix = np.zeros([len(params), n])
    for idx, key in enumerate(params):
        f, t = params[key]
        np.clip(np.linspace(f, t, n, endpoint=True), f, t, out=property_matrix[idx])

    displacement = (np.random.rand(n - 2) - .5) * 2 * d
    property_matrix[:, 1:-1] -= property_matrix[:, 1:-1] * displacement

    index_vector = np.zeros(len(params), dtype='int64')
    while True:
        properties = HyperSpace()
        for idx, key in enumerate(params):
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

            if k == len(params):
                return

def hyper_space(params={'lr': [5e-8, 5e-1]}, n=5, d=0):
    property_matrix = np.zeros([len(params), n])
    for idx, key in enumerate(params):
        f, t = params[key]
        np.clip(np.linspace(f, t, n, endpoint=True), f, t, out=property_matrix[idx])

    displacement = (np.random.rand(n - 2) - .5) * 2 * d
    property_matrix[:, 1:-1] -= property_matrix[:, 1:-1] * displacement

    index_vector = np.zeros(len(params), dtype='int64')
    while True:
        properties = HyperSpace()
        for idx, key in enumerate(params):
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

            if k == len(params):
                return


class BaseAgent:
    def __init__(self, conf):
        self.conf = conf

    def reset(self):
        """ """

    def act(self, state):
        """ """

    def step(self, state, action, reward, next_state, done):
        """ """

    def learn(self, experiences):
        """ """

    def save(self, path, name):
        """ """


class SharedMemAgent(BaseAgent):
    def __init__(self, conf, agents, update_every_n_steps=None):
        super().__init__(conf)
        self.memory = ReplayBuffer(conf.buffer_size, conf.s_dim, conf.a_dim, conf.seed)
        self.agents = agents
        self.update_every_n_steps = update_every_n_steps
        self.step_n = 0

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states):
        return list(a.act(s) for a, s in zip(self.agents, states))

    def step(self, states, actions, rewards, next_states, dones):
        self.step_n += 1
        for idx in range(len(self.agents)):
            self.memory.add(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])

        if not self.update_every_n_steps:
            self.try_learn()
        elif self.step_n > self.update_every_n_steps:
            self.try_learn()
            self.step_n = 0

    def try_learn(self):
        if len(self.memory) > self.conf.batch_size:
            for agent in self.agents:
                sample = self.memory.sample(self.conf.batch_size)
                sample = torch.from_numpy(sample).to(self.conf.device)
                if isinstance(agent, ddpg_agent.Agent):
                    sample = unmap(sample, self.conf.s_dim, self.conf.a_dim)
                agent.learn(sample)

    def learn(self, experiences):
        """ NoOp """

    def __str__(self):
        return 'SharedMemory{' + ';'.join([str(agent) for agent in self.agents]) + '}'


def multi_ddpg_with_shared_mem(conf, n):
    conf.mem_disabled = True
    conf.seed = None
    return SharedMemAgent(conf, list([ddpg.Agent(conf) for _ in range(n)]))
