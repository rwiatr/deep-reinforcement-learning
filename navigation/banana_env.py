from collections import deque

from drl.dql.agent import Agent
from drl.benchmark import ScorePlot
import numpy as np


class EnvHelper:

    def __init__(self, env, brain_name, score_plot=ScorePlot()):
        self.env = env
        self.brain_name = brain_name
        self.score_plot = score_plot
        self.env_info = None
        self.state = None
        self.scores = None
        self.agent = None
        self.name = None
        self.eps = None
        self.all_scores = None
        self.best_average = None

    def set_agent(self, agent: Agent, name=None, eps_start=1.0):
        self.agent = agent
        self.name = name if name else str(agent)
        self.scores = deque(maxlen=100)  # last 100 scores
        self.all_scores = []
        self.eps = eps_start
        self.best_average = np.NINF

    def load(self, name=None):
        self.agent.load('saved', name)

    def run(self, episodes=1000, train_mode=True, fast_mode=True, step_limit=None, eps_end=0.01, eps_decay=0.995,
            eps_step=None):
        print('start with train_mode={}'.format(train_mode))
        self.eps = 1 if train_mode else 0

        for episode in range(episodes):
            score = 0  # initialize the score
            self.env_info = self.env.reset(train_mode=fast_mode)[self.brain_name]  # reset the environment
            self.state = self.env_info.vector_observations[0]

            if len(self.scores) == 0:
                np_mean = np.NINF
                np_min = np.NAN
            else:
                np_mean = np.mean(self.scores)
                np_min = np.min(self.scores)

            if (episode + 1) % 100 == 0:
                self.print_stats(episode, np_mean, np_min, score)

            step = 0

            while True:
                killed = False
                action = self.agent.act(self.state, self.eps)

                self.env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = self.env_info.vector_observations[0]  # get the next state
                reward = self.env_info.rewards[0]  # get the reward
                done = self.env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score
                if train_mode:
                    self.agent.step(state=self.state,
                                    action=action,
                                    reward=reward,
                                    next_state=next_state,
                                    done=done)

                self.state = next_state  # roll over the state to next time step

                step += 1
                if step_limit and step >= step_limit:
                    self.print_stats(episode, np_mean, np_min, score, "Step limit exceeded")

                    killed = True
                    score -= 1000

                if done or killed:  # exit loop if episode finished
                    self.print_stats(episode, np_mean, np_min, score, "Done")
                    self.scores.append(score)
                    self.all_scores.append(score)
                    self.score_plot.add(self.name, self.all_scores)
                    if self.best_average < np_mean:
                        self.best_average = np_mean
                        self.agent.save('saved')
                    self.best_average = np.max([self.best_average, np_mean])
                    break
            if eps_step:
                self.eps = max(eps_end, self.eps - eps_step)
            else:
                self.eps = max(eps_end, eps_decay * self.eps)  # decrease epsilon

        print('\n{} Best Average Score {:.2f}'.format(self.name, self.best_average))

    def print_stats(self, episode, np_mean, np_min, score, status=None):
        if status:
            print('\rEpisode {} ## Average Score: {:.2f} ## Best Average Score: {:.2f}'
                  ' ## Last Min Score: {:.2f} ## Eps: {:.2f} ## Score: {} ## {}\t\t\t'
                  .format(episode + 1, np_mean, self.best_average, np_min, self.eps, score, status),
                  end="")
        else:
            print('\rEpisode {} ## Average Score: {:.2f} ## Best Average Score: {:.2f}'
                  ' ## Last Min Score: {:.2f} ## Eps: {:.2f} ## Score: {}\t\t\t'
                  .format(episode + 1, np_mean, self.best_average, np_min, self.eps, score),
                  end="")

    def show_plot(self, mode=None):
        self.score_plot.show(mode=mode)
