from os.path import expanduser
from drl.env import EnvHelper
from drl.unity.env import SingleBrainEnv
import drl.hyperparams as hp
import drl.agent.DDPG as ddpg
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# env = SingleBrainEnv(expanduser("~") + '/Reacher_Linux_NoVis_one_agent/Reacher.x86_64')
env = SingleBrainEnv(expanduser("~") + '/Reacher_multi.app')
helper = EnvHelper(env)
parameters = {
    'seed': [311, 13],
    'batch_size': [128, 512, 1024],
    'lr_c': [1e-6, 1e-4, 1e-3],
    'lr_a': [1e-6, 1e-4, 1e-3],
    'wd_a': [0, 0.001],
    'wd_c': [0, 0.001]
}

for idx, conf in enumerate(hp.hyper_space_ns(parameters)):
    print('%d --------------------------------------------------------------' % idx)
    conf.s_dim = 33
    conf.a_dim = 4
    print(conf)
    helper.set_agents(ddpg.Agent(conf, device, env.get_num_agents()))
    helper.run_until(episodes=60, print_every=60)
    print('----------------------------------------------------------------')
