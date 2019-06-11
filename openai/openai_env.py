import gym
from collections import deque

from drl.benchmark import ScorePlot
from drl.pbm.agent import Agent
import numpy as np


class EnvHelper:

    def __init__(self, score_plot=ScorePlot(), seed=101, env_name='MountainCarContinuous-v0'):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        np.random.seed(seed)
        self.score_plot = score_plot
        self.agent = None
        self.name = None

        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)

    def set_agent(self, agent: Agent, name=None):
        self.agent = agent
        self.name = name if name else str(agent)

    def load(self, name=None):
        self.agent.load('saved', name)

    def run_until(self, episodes=None, target_mean_reward=float('inf'), print_every=1):
        episode = 1
        scores_deque = deque(maxlen=100)
        scores = []
        while True:
            reward = self.agent.train_epoch(self.env)
            scores_deque.append(reward)
            scores.append(reward)

            if episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

            if episode == episodes or np.mean(scores_deque) >= target_mean_reward:
                break
            episode += 1

        self.score_plot.add(self.name, scores)

        if np.mean(scores_deque) >= 90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                           np.mean(scores_deque)))

    def show_plot(self, mode=None):
        self.score_plot.show(mode=mode)

    def show_sim_notebook(self):
        import matplotlib.pyplot as plt
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1400, 900))
        display.start()

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()

        state = self.env.reset()
        img = plt.imshow(self.env.render(mode='rgb_array'))
        while True:
            action = self.agent.step_no_grad(state)
            img.set_data(self.env.render(mode='rgb_array'))
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if done:
                break
