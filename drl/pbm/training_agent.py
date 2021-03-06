import torch.nn as nn
import torch
import math
import numpy as np
import drl.pbm.agent as pbm
import torch.nn.functional as F


class TrainingAgentProperties:
    def __init__(self,
                 pop_size=50,
                 gamma=1.0,
                 max_t=1000,
                 n_elite=None,
                 elite_fac=0.2,
                 sigma=0.5):
        self.pop_size = pop_size
        self.gamma = gamma
        self.max_t = max_t
        self.n_elite = int(n_elite if n_elite else elite_fac * pop_size)
        self.sigma = sigma

    def __str__(self):
        return "pop_size={};gamma={};max_t={};n_elite={};sigma={}" \
            .format(self.pop_size, self.gamma, self.max_t, self.n_elite, self.sigma)


class TrainingAgent(pbm.Agent):

    def __init__(self, model, device, properties=TrainingAgentProperties(), best_weight=None):
        self.properties = properties
        self.model = model.to(device)
        self.device = device
        self.pop_size = properties.pop_size
        self.gamma = properties.gamma
        self.max_t = properties.max_t
        self.n_elite = properties.n_elite
        self.sigma = properties.sigma
        self.best_weight = best_weight \
            if best_weight \
            else self.sigma * np.random.randn(self.model.get_weights_dim())

    def evaluate(self, env, weights, gamma=1.0):
        self.model.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        for t in range(self.max_t):
            action = self.do_action(state)
            state, reward, done, _ = env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

    def do_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.model.forward(state)
        return action

    def train_epoch(self, env):
        weights_pop = [self.best_weight + (self.sigma * np.random.randn(self.model.get_weights_dim()))
                       for _ in range(self.pop_size)]
        rewards = np.array([self.evaluate(env, weights, self.gamma) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-self.n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        self.best_weight = np.array(elite_weights).mean(axis=0)
        self.save('saved')

        return self.evaluate_best(env)

    def evaluate_best(self, env):
        return self.evaluate(env, self.best_weight, gamma=1.0)

    def save(self, directory, file='checkpoint.pth'):
        torch.save(self.model.state_dict(), directory + '/' + file + '.pth')

    def load(self, directory, file='checkpoint.pth'):
        self.model.load_state_dict(torch.load(directory + '/' + file + '.pth'))

    def __str__(self):
        return "training_agent::{}".format(str(self.properties))

    def step_no_grad(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.model(state)
        return action


class Model(nn.Module):
    def __init__(self, env, h_size=16):
        super(Model, self).__init__()
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        import gym
        if type(env.action_space) is gym.spaces.discrete.Discrete:
            self.a_size = env.action_space.n
            self.discrete_action_space = True
        else:
            self.a_size = env.action_space.shape[0]
            self.discrete_action_space = False
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))

        if self.discrete_action_space:
            return x.argmax().cpu().data.item()

        return x.cpu().data

    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size
