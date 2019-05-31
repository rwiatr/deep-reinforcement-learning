import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from drl.pbm.training_agent import TrainingAgent, TrainingAgentProperties, Model
from openai.openai_env import EnvHelper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

helper = EnvHelper(env_name="CartPole-v0")

TRAIN_EPISODES = 400
TEST_EPISODES = 400
DEMO_EPISODES = 5

model_a = Model(helper.env)
# model_b = Model(helper.env, h_size=32)
agent_a = TrainingAgent(model_a, device)
# agent_b = TrainingAgent(model_b, device)

helper.set_agent(agent_a)
helper.run_until(episodes=TRAIN_EPISODES)

# helper.set_agent(agent_b)
# helper.run_until(episodes=TRAIN_EPISODES)
