from unityagents import UnityEnvironment
from os.path import expanduser

from continuous_control.cc_env import EnvAccessor, EnvHelper, EnvHelperMultiAgent, EnvHelperMultiAgent2, \
    OpenAiEnvAccessor
from drl.agent.SharedMem import multi_ddpg_with_shared_mem
from drl.agent.utils import OUNoise
import drl.agent.original.ddpg_agent as ddpg_agent
import drl.agent.DDPG as ddpg

home = expanduser("~")
path = home + '/Reacher_multi.app'
path = home + '/Reacher.app'

print(path)
# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name=path)




class Object:
    pass


conf = Object()
conf.buffer_size = int(1e5)
conf.seed = 1
# conf.s_dim = 33
# conf.a_dim = 4
conf.s_dim = 3
conf.a_dim = 1
conf.lr_c = 1e-3
conf.lr_a = 1e-4
conf.noise = OUNoise(conf.a_dim, conf.seed).reset()
conf.device = 'cpu'
conf.batch_size = 128
conf.gamma = 0.99
conf.tau = 1e-3
conf.max_t = 30

accessor = OpenAiEnvAccessor('Pendulum-v0')
# accessor = EnvAccessor(env)
# accessor.set_train_mode(True)
helper = EnvHelper(accessor)
# helper = EnvHelperMultiAgent2(accessor)
# agent = ddpg.Agent(conf)
agent = ddpg_agent.Agent(conf)
# agent = multi_ddpg_with_shared_mem(conf, 1)
helper.set_agent(agent)
helper.run_until(1000, print_every=10)

env.close()