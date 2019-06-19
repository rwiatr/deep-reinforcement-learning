import torch.nn as nn
import torch
import math
import numpy as np
import drl.pbm.agent as pbm
import torch.nn.functional as F


class ReinforceAgentProperties:
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


class ReinforceAgent:

    def __init__(self, model, device, properties=ReinforceAgentProperties(), best_weight=None):
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


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        ########
        ##
        ## Modify your neural network
        ##
        ########

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # output = 20x20 here
        self.conv = nn.Conv2d(2, 1, kernel_size=4, stride=4)
        self.size = 1 * 20 * 20

        # 1 fully connected layer
        self.fc = nn.Linear(self.size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        ########
        ##
        ## Modify your neural network
        ##
        ########

        x = F.relu(self.conv(x))
        # flatten the tensor
        x = x.view(-1, self.size)
        return self.sig(self.fc(x))