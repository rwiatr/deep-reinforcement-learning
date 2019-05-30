import numpy as np

from drl.dql.agent import Agent


class RandomAgent(Agent):
    def __init__(self, state_size, action_size):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, eps=0.):
        return np.random.randint(self.action_size)

    def step(self, state, action, reward, next_state, done):
        """"""
