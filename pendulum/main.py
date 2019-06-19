from collections import deque

import gym

import drl.agent.DDPG as ddpg
from drl.agent.utils import OUNoise
from pendulum.pendulum_env import EnvHelper

env = gym.make('Pendulum-v0')
env.seed(2)


class Object:
    pass


conf = Object()
conf.buffer_size = int(1e5)
conf.seed = 1
conf.s_dim = 3
conf.a_dim = 1
conf.lr_c = 1e-3
conf.lr_a = 1e-4
conf.noise = OUNoise(conf.a_dim, conf.seed).reset()
conf.device = 'cpu'
conf.batch_size = 128
conf.gamma = 0.99
conf.tau = 1e-3
conf.max_t = 300

agent = ddpg.Agent(conf)

helper = EnvHelper('Pendulum-v0')
helper.set_agent(agent)
helper.run_until(episodes=1000)
helper.show_plot(mode='average=100')

env.close()
