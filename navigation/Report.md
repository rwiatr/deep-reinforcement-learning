# Udacity DRLND Project 01 Report

## Introduction

This report is a part of [this repository](https://github.com/rwiatr/deep-reinforcement-learning) and was created for 
the [Navigation](https://github.com/rwiatr/deep-reinforcement-learning/blob/master/navigation/README.md) project.

## Learning Algorithm

This task was solved using a Deep Q-Network (DQN).
DQN was implemented using two fully connected layers with ReLU activations 
and one output layer with linear output.
While the agent was interacting with the environment the state was
ran through network A and were translated to an action.
Techniques used in the agent:
 - &epsilon; greedy action selection for exploration
 - state replay for gathering experience of the agent and using it in learning 
 - fixed Q targets controlled by tau
 
In essence the algorithm has two steps:
 1) Sampling the environment and storing results in the buffer
 2) Learning step based on data sampled from the buffer

### Neural Network Architecture
For this task I have chosen a three layer neural network 
with ReLu activation function on the first two layers. 
All layers have the same width which is a hyperparameter.

## Result
It depending on the run it can take as little as 250 epochs to train an agent that is able to solve the 
environment achieving a average reward of 13+ over last 100 epochs and 270 epochs to get to 14+. 

![400 iterations averaged over 100 steps](result.png)

| Agent | target &epsilon; | &epsilon; step |
|:-------------|:-------------|:-----|
| A | 0.05 | 0.01 |
| B | 0.05 | 0.05 |
| C | 0.01 | 0.01 |
| D | 0.01 | 0.05 |
| E | 0.01 | 1.0 |

&epsilon; step of 0.05 means that &epsilon; will reach minimum after 20 steps.
The results show that agents having lower target &epsilon; achieve better final score.
This can be explained by the fact that less randomness in the final stages allows to exploit
the policy of the agent.
The results show also that faster bigger &epsilon; step is positive on learning rate.
This may mean the environment is simple, repetitive and requires limited exploration.

### Hyperparameters
Hyperparameters were generated with using random sampling of hyperparameters space and
then I selected hyperparameters that gave the best effect:

| Name | Value | Description |
|:-------------|:-------------|:-----|
| buffer_size | 1e5 | Size of the memory buffer for storing events |
| batch_size | 64 | Batch size for training the network  |
| gamma | 0.99 | discount factor |
| tau | 1e-3 | network interpolation parameter |
| lr | 5e-4 | Learning rate |
| update_every | 4 | How many steps should the agent take before the training will be executed |
| fc_size | 37 | Number of neurons in each hidden layer |


## Future work
 - [ ] Prioritize experience replay
 - [ ] Implement CNN as input
 - [ ] Implement [Rainbow](https://arxiv.org/abs/1710.02298)
