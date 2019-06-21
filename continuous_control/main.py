from unityagents import UnityEnvironment
from os.path import expanduser

from continuous_control.cc_env import EnvAccessor, EnvHelper
from drl.agent.utils import OUNoise
import drl.agent.DDPG as ddpg

home = expanduser("~")
path = home + '/Reacher.app'
print(path)
# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name=path)

accessor = EnvAccessor(env)
helper = EnvHelper(accessor)


class Object:
    pass


conf = Object()
conf.buffer_size = int(1e5)
conf.seed = 1
conf.s_dim = 33
conf.a_dim = 4
conf.lr_c = 1e-3
conf.lr_a = 1e-4
conf.noise = OUNoise(conf.a_dim, conf.seed).reset()
conf.device = 'cpu'
conf.batch_size = 128
conf.gamma = 0.99
conf.tau = 1e-3
conf.max_t = 300
conf.a = False
agent = ddpg.Agent(conf)
helper.set_agent(agent)
helper.run_until(1000, print_every=10)

env.close()
