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
helper.set_agents(ddpg.Agent(hp.default(), device, env.get_num_agents()))
helper.run_until(episodes=100)
