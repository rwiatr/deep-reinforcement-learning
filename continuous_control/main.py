import gym
import torch

import drl.agent.DDPG as ddpg
import drl.agent.base
import drl.agent.original.ddpg_agent as ddpg_agent
from drl.agent.utils import OUNoise
from pendulum.pendulum_env import EnvHelper
from unityagents import UnityEnvironment
from os.path import expanduser

from continuous_control.cc_env import EnvAccessor, EnvHelper, EnvHelperMultiAgent2, \
    OpenAiEnvAccessor, OpenAiEnvAccessorMulti

from drl.agent.utils import OUNoise
import drl.agent.original.ddpg_agent as ddpg_agent
import drl.agent.DDPG as ddpg
import drl.agent.base as base

for conf in base.hyper_space(params={'batch_size': [32, 512]}, n=4):
    conf.buffer_size = int(1e5)
    conf.seed = 1
    conf.s_dim = 33
    conf.a_dim = 4
    conf.wd_a = 0
    conf.wd_c = 0.001
    # conf.s_dim = 3
    # conf.a_dim = 1
    conf.lr_c = 1e-3
    conf.lr_a = 1e-4
    conf.noise = OUNoise(conf.a_dim, conf.seed).reset()
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.batch_size = 128
    conf.gamma = 0.99
    conf.tau = 1e-3
    conf.max_t = 30000
    conf.mem_disabled = False

    # accessor = OpenAiEnvAccessor('Pendulum-v0')
    # helper = EnvHelper(accessor)
    # agent = ddpg.Agent(conf)
    # agent = ddpg_agent.Agent(conf)

    # accessor = OpenAiEnvAccessorMulti('Pendulum-v0')
    # agent = base.SharedMemAgent(conf, [ddpg_agent.Agent(conf)])
    agent = ddpg_agent.Agent(conf)
    # agent = sm.Agent(conf, [ddpg.Agent(conf)])

    home = expanduser("~")
    path = home + '/Reacher_multi.app'
    path = home + '/Reacher.app'
    env = UnityEnvironment(file_name=path)
    accessor = EnvAccessor(env)

    accessor.set_train_mode(True)
    # agent = multi_ddpg_with_shared_mem(conf, 1)
    helper = EnvHelperMultiAgent2(accessor)
    print(agent)
    helper.set_agent(agent)
    helper.run_until(1000, print_every=5)
