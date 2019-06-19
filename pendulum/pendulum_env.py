from collections import deque

import gym

from drl.agent.base import BaseAgent
from drl.benchmark import ScorePlot
import numpy as np


class EnvHelper:

    def __init__(self, env_name, score_plot=ScorePlot(), seed=101):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        np.random.seed(seed)
        self.score_plot = score_plot
        self.agent = None
        self.name = None

        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)

    def set_agent(self, agent: BaseAgent, name=None):
        self.agent = agent
        self.name = name if name else str(agent)

    def load(self, name=None):
        self.agent.load('saved', name)

    def run_until(self, episodes=None, target_mean_reward=float('inf'), print_every=1):
        episode = 1
        scores_deque = deque(maxlen=100)
        scores = []
        while True:
            state = self.env.reset()
            if self.agent.conf.noise:
                self.agent.conf.noise.reset()
            score = 0
            for t in range(self.agent.conf.max_t):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_deque.append(score)
            scores.append(score)

            if episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

            if episode == episodes or np.mean(scores_deque) >= target_mean_reward:
                break
            episode += 1

        self.score_plot.add(self.name, scores)

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
