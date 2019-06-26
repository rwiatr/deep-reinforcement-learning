import numpy as np
from collections import deque
from drl.benchmark import ScorePlot


class BaseAgent:
    def __init__(self, conf):
        self.conf = conf

    def reset(self):
        """ """

    def act(self, state):
        """ """

    def step(self, state, action, reward, next_state, done):
        """ """

    def learn(self, experiences):
        """ """

    def save(self, path, name):
        """ """


class Env:
    def reset(self):
        """ resets the environment """
        raise NotImplemented()

    def step(self, actions):
        """ makes one step in the environment by n agents denoted by actions """
        raise NotImplemented()

    def get_num_agents(self):
        """ returns the number of agents living in environment """
        raise NotImplemented()

    def get_action_dim(self):
        raise NotImplemented()

    def get_state_dim(self):
        raise NotImplemented()

    def close(self):
        """ closes the environment """
        raise NotImplemented()


class EnvHelper:

    def __init__(self, env: Env, score_plot=ScorePlot(), save_best=False):
        self.env = env
        self.score_plot = score_plot
        self.save_best = save_best
        self.agents = None
        self.name = None

    def set_agents(self, agents: BaseAgent, name=None):
        self.agents = agents
        self.name = name if name else str(agents)

    def load(self, name=None):
        self.agents.load('saved', name)

    def run_until(self, episodes=None, max_t=None, target_mean_reward=float('inf'), print_every=1,
                  depths=(100, 10)):
        episode = 1
        scores_a_deque = deque(maxlen=depths[0])
        scores_b_deque = deque(maxlen=depths[1])
        scores = []
        max_mean = -float('inf')

        while True:
            state = self.env.reset()
            self.agents.reset()
            score = np.zeros(self.env.get_num_agents())
            steps = 0

            while True:
                action = self.agents.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agents.step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                steps += 1

                if np.any(done):
                    break

                if max_t and steps >= max_t:
                    break

            mean_score = np.mean(score)
            scores_a_deque.append(mean_score)
            scores_b_deque.append(mean_score)
            scores.append(mean_score)

            pattern = '\rEpisode {}\tAverage Score {:d}: {:.2f}\tAverage Score {:d}: {:.2f}\tScore: {:.2f}\tSteps: {:d}'
            print(pattern.format(episode, depths[0], np.mean(scores_a_deque),
                                 depths[1], np.mean(scores_b_deque), mean_score, steps), end="")
            if episode % print_every == 0:
                print(pattern.format(episode, depths[0], np.mean(scores_a_deque),
                                     depths[1], np.mean(scores_b_deque), mean_score, steps))

            if max_mean < np.mean(scores_b_deque):
                max_mean = np.mean(scores_b_deque)
                if self.save_best:
                    self.agents.save("saved", self.name)

            if episode == episodes or np.mean(scores_a_deque) >= target_mean_reward:
                break

            episode += 1

        self.score_plot.add(self.name, scores)

    def show_plot(self, mode=None):
        self.score_plot.show(mode=mode)
