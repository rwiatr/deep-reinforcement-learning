from os.path import expanduser
from drl.env import EnvHelper
from drl.unity.env import SingleBrainEnv

env = SingleBrainEnv(expanduser("~") + '/Reacher_Linux_NoVis_one_agent/Reacher.x86_64')
helper = EnvHelper(env)

helper.run_until(episodes=100)
