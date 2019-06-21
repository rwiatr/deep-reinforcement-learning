from collections import deque

import gym

from drl.agent.base import BaseAgent
from drl.benchmark import ScorePlot
import numpy as np


class EnvAccessor:
    def __init__(self, env):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.train_mode = True

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]  # reset the environment
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        return (env_info.vector_observations, env_info.rewards, env_info.local_done, None)

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode


class OpenAiEnvAccessor:
    def __init__(self, name, seed=1010):
        self.env = gym.make(name)
        self.env.seed(seed)
        np.random.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)


class OpenAiEnvAccessorMulti:
    def __init__(self, name, seed=1010):
        self.env = gym.make(name)
        self.env.seed(seed)
        np.random.seed(seed)

    def reset(self):
        return [self.env.reset()]

    def step(self, actions):
        next_state, reward, done, rest = self.env.step(actions[0])
        return [next_state], [reward], [done], [rest]


class EnvHelper:

    def __init__(self, env, score_plot=ScorePlot(), seed=101, ):
        self.env = env
        np.random.seed(seed)
        self.score_plot = score_plot
        self.agent = None
        self.name = None

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

            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, np.mean(scores_deque), score),
                  end="")
            if episode % print_every == 0:
                # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

            if episode == episodes or np.mean(scores_deque) >= target_mean_reward:
                break
            episode += 1

        self.score_plot.add(self.name, scores)

    def show_plot(self, mode=None):
        self.score_plot.show(mode=mode)

    # def show_sim_notebook(self):
    #     import matplotlib.pyplot as plt
    #     from pyvirtualdisplay import Display
    #     display = Display(visible=0, size=(1400, 900))
    #     display.start()
    #
    #     is_ipython = 'inline' in plt.get_backend()
    #     if is_ipython:
    #         from IPython import display
    #
    #     plt.ion()
    #
    #     state = self.env.reset()
    #     img = plt.imshow(self.env.render(mode='rgb_array'))
    #     while True:
    #         action = self.agent.step_no_grad(state)
    #         img.set_data(self.env.render(mode='rgb_array'))
    #         plt.axis('off')
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #         next_state, reward, done, _ = self.env.step(action)
    #         state = next_state
    #         if done:
    #             break


class EnvHelperMultiAgent:

    def __init__(self, env: EnvAccessor, score_plot=ScorePlot(), seed=101, n_agents=1):
        self.env = env
        self.n_agents = n_agents
        np.random.seed(seed)
        self.score_plot = score_plot
        self.agent = None
        self.name = None

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
                action = list([self.agent.act(s) for s in state])
                next_state, reward, done, _ = self.env.step(action)

                for idx in range(self.n_agents):
                    self.agent.step(state[idx], action[idx], reward[idx], next_state[idx], done[idx])

                state = next_state
                score += np.mean(reward)
                if np.any(done):
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


class EnvHelperMultiAgent2:

    def __init__(self, env: EnvAccessor, score_plot=ScorePlot(), seed=101):
        self.env = env
        np.random.seed(seed)
        self.score_plot = score_plot
        self.agent = None
        self.name = None

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
                score += np.mean(reward)
                if np.any(done):
                    break

            scores_deque.append(score)
            scores.append(score)

            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, np.mean(scores_deque), score),
                  end="")
            if episode % print_every == 0:
                # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

            if episode == episodes or np.mean(scores_deque) >= target_mean_reward:
                break

            episode += 1

        self.score_plot.add(self.name, scores)

    def show_plot(self, mode=None):
        self.score_plot.show(mode=mode)
