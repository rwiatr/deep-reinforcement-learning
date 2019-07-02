# Collaboration and Competition

You can find all the details of the project goals and environment [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet/README.md). 
This module contains an implementation of an agent capable of learning how to behave in this environment.
To run this code simply clone this repository, download the Unity environment from [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet/README.md) 
and start a jupyter notebook and execute [this script](collaboration_and_competition.ipynb).
Agent weights are saved [here](saved).

### Environment Details

Tennis game is a two-player game where agents control rackets to bounce ball over a net.
The goal for the agents is to bounce ball between one another while not dropping or sending ball out of bounds.
Both agents are linked to a single Brain named TennisBrain.
Agent Reward Function (independent):
- +0.1 To agent when hitting ball over net.
- -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
Observation and action space (each agent receives its own, local observation):
- Vector Observation space: 8 variables corresponding to position and velocity of ball and racket.
- Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
- Visual Observations: None.

The task is episodic. It is considered to be solved when the agents get and average score of +0.5
 over 100 consecutive episodes, after taking the maximum over both agents.

Some more details (for example how to add an extra brain) are available [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)

### Getting Started

Download the Unity environment from [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/p3_collab-compet/README.md#getting-started) 
[setup.py](/setup.py) contains all the python packages that need to be installed before running the script.

### Instructions

Start a jupyter notebook and execute [this script](collaboration_and_competition.ipynb).
The script will attempt to train several agents and it saves the best parameters of each agent.