from unityagents import UnityEnvironment
import numpy as np
from os.path import expanduser
home = expanduser("~")
path = home + '/Reacher.app'
env = UnityEnvironment(file_name=path)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])



##########

import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline
from unityagents import UnityEnvironment
from os.path import expanduser


from bkp.ddpg_agent import Agent
from os.path import expanduser
def ddpg(env, brain_name, agent, n_episodes=2000, max_t=1001):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step([action])[brain_name]
            next_state, reward, done, _ = (env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0], None)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
    return scores

##########

agent = Agent(state_size=33, action_size=4, random_seed=10)

scores = ddpg(env, brain_name, agent, n_episodes=50)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
