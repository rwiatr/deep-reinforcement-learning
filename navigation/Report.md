# Udacity DRLND Project 01 Report

## Introduction

This report is a part of [this repository](https://github.com/rwiatr/deep-reinforcement-learning) and was created for 
the [Navigation](https://github.com/rwiatr/deep-reinforcement-learning/blob/master/navigation/README.md) project.

## Method

## Result
It took only 300 epochs to train an agent that is able to solve the 
environment achieving a average reward of 13+ over last 100 epochs. 

| Name | Value | Description |
|:-------------|:-------------|:-----|
| buffer_size | 1e5 | Size of the memory buffer for storing events |
| batch_size | 64 | Batch size for training the network  |
| gamma | 0.99 | |
| tau | 1e-3 | |
| lr | 5e-4 | Learning rate |
| update_every | 4 | How many steps should the agent take before the training will be executed |
| fc_size | 64 | Number of neurons in each hidden layer |

## Future work
 - [ ] Implement CNN as input
 - [ ] Implement [Rainbow](https://arxiv.org/abs/1710.02298)
