import numpy as np
from collections import deque

from drl.agent.base import BaseAgent
from drl.benchmark import ScorePlot


class Env:
    def reset(self):
        """ resets the environment """

    def step(self, actions):
        """ makes one step in the environment by n agents denoted by actions """
        return (None, None, None, None)

    def num_agents(self):
        """ returns the number of agents living in environment"""
        return 0


class EnvHelper:

    def __init__(self, env: Env, score_plot=ScorePlot()):
        self.env = env
        self.score_plot = score_plot
        self.agents = None
        self.name = None

    def set_agents(self, agents: BaseAgent, name=None):
        self.agents = agents
        self.name = name if name else str(agents)

    def load(self, name=None):
        self.agents.load('saved', name)

    def run_until(self, episodes=None, max_t=int('inf'), target_mean_reward=float('inf'), print_every=1,
                  depths=(100, 10)):
        episode = 1
        scores_a_deque = deque(maxlen=depths[0])
        scores_b_deque = deque(maxlen=depths[1])
        scores = []
        max_mean = -float('inf')

        while True:
            state = self.env.reset()
            self.agents.reset()
            score = np.zeros(self.env.num_agents())
            steps = 0

            for t in range(max_t):
                action = self.agents.act(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                score += reward
                steps += 1

                if np.any(done):
                    break

            scores_a_deque.append(score)
            scores_b_deque.append(score)

            scores.append(np.mean(score))

            print('\rEpisode {}\tAverage Score ({:d}): {:.2f}'
                  '\tAverage Score ({:d}): {:.2f}\tScore: {:.2f}\tSteps: {:d}'
                  .format(episode,
                          depths[0], np.mean(scores_a_deque),
                          depths[1], np.mean(scores_b_deque),
                          score, steps), end="")
            if episode % print_every == 0:
                print(
                    '\rEpisode {}\tAverage Score ({:d}): {:.2f}'
                    '\tAverage Score ({:d}): {:.2f}\tScore: {:.2f}\tSteps: {:d}'
                        .format(episode,
                                depths[0], np.mean(scores_a_deque),
                                depths[1], np.mean(scores_b_deque),
                                score, steps))

            if max_mean < np.mean(scores_b_deque):
                max_mean = np.mean(scores_b_deque)
                self.agents.save("saved", self.name)

            if episode == episodes or np.mean(scores_a_deque) >= target_mean_reward:
                break

            episode += 1

        self.score_plot.add(self.name, scores)

    def show_plot(self, mode=None):
        self.score_plot.show(mode=mode)
