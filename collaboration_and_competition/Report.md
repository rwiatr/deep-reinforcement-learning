# Udacity DRLND Project 03 Report

## Introduction

This report is a part of [this repository](https://github.com/rwiatr/deep-reinforcement-learning) and was created for 
the [Collaboration and Competition](https://github.com/rwiatr/deep-reinforcement-learning/blob/master/collaboration_and_competition/README.md) project.

## Learning Algorithm

This task was solved using [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971).
DDPG learns a Q-function and uses it to learn a policy &pi;. 
A detailed description can be found [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).
DDPG can be thought of as being deep Q-learning for continuous action space.

### Neural Network Architecture
Both, Critic and Actor design was based on the [original paper](https://arxiv.org/abs/1509.02971).
For the Actor I used a fully connected network with two layers: 24x256 and 256x2. 
For the Critic I used a fully connected network with five layers: 24x256, 256(+2 action)x256, 256x128 and 128x1.
## Result
It took 1571 episodes to achieve the score of 51. The agent was cut off after reaching this target.
![](result.png)

During the experiment I found that it is challenging to predict if an agent will be able to start learning.
In some cases it took 2000 iterations to train an agent while in other cases it took over 5000.
### Hyperparameters
Hyperparameters were generated with using sampling of hyperparameters space and
then I selected hyperparameters that gave the best effect:

| Name | Value | Description |
|:-------------|:-------------|:-----|
| buffer_size | 1e5 | Size of the memory buffer for storing events |
| batch_size | 512 | Batch size for training the network  |
| gamma | 0.99 | discount factor |
| tau | 1e-3 | network interpolation parameter |
| lr_a | 1e-4 | Actor learning rate |
| lr_c | 1e-4 | Critic learning rate |

## Future work
 - [ ] Prioritize experience replay
 - [ ] Implement CNN as input
 - [ ] Make training more predictable
 - [ ] Implement MADDPG