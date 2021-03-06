# Deep Reinforcement Learning Nanodegree
# *DRLND-MARL-Collaboration-and-Competition*

## Introduction

In this project, two agents control rackets to bounce a ball over a net (Unity's Tennis environment).

<p align="center">
<img src="Images/Tennis_3.png" height=300 />
</p>

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
    
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
    
The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) will be add up, to get a score for each agent. This yields 2 (potentially different) scores. From them, the maximum of these 2 scores is taken.
- This yields a single score for each episode.
    
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
    
### Getting Started

Setup the dependencies as described [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md).

Download the environment from one of the links below.
You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zipp)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)
    
    
Clone the repository and unpack the environment file in the project folder.

### Instructions

Start `Tennis.ipynb` in jupyter notebook:
```
jupyter notebook Tennis.ipynb &
```
Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

