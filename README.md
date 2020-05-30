[p3_solution]: images/p3_solution.gif "p3_solution"

# Udacity Reinforcement Learning Nanodegree Project Three: Collaboration and Competition

This is my submission for Project Three: Collaboration and Competition.

Code is based on https://github.com/ShangtongZhang/DeepRL
 hash [5d0ad07](https://github.com/ShangtongZhang/DeepRL/commit/5d0ad07c7f2081123fddc4faf8db2aa09730e85b) with the following improvements:

* removed dependency on OpenAI Baselines
* adapted to use Unity ML-Agents
* Twined Delayed DDPG (TD3) has been adapted to use environemts containing multiple agents

See Report.md for more details

## Tennis Environment

This submission solves the Tennis environment.

![Tennis Environment][p3_solution]

The goal of this environment ....

The observation space ....

This version of the Tenis environment is considered solved when an average score of xxx

## Getting Started

1. Download this codebase

2. Download the Tenis environment

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the same folder as the codebase and unzip the file

3. To install and create the Conda environment

``` bash
conda env create -f environment.yml
conda activate p3
pip install -r requirements.txt
cd unityagents
pip install -e .
cd ..
pip install -e .
```

## Instructions

### To train the agent

``` bash
python train.py
```

To follow training via tensorboard

``` bash
tensorboard --logdir=tf_log
```

### To view the pre-trained agent

``` bash
python play.py
```

## References

* [Modularized Implementation of Deep RL Algorithms in PyTorch](https://github.com/ShangtongZhang/DeepRL)
