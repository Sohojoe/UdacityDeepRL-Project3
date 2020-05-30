[average_over_3_runs]: images/average_over_3_runs.png "average_over_3_runs"
[episode_returns_ave100]: images/episode_returns_ave100.png "episode_returns_ave100"
[p2_solution]: images/p2_solution.gif "p2_solution"
[raw_average_over_3_runs]: images/raw_average_over_3_runs.png "raw_average_over_3_runs"
[raw_episode_returns]: images/raw_episode_returns.png "raw_episode_returns"
[trained model]: images/trained_model.png "trained model"

# Udacity Reinforcement Learning Nanodagree - Project Two

Joe Booth April 2019

## Project Objective

The goal of this project is to solve the "Tennis" environment using Twined Delayed DDPG (TD3) (Fujimoto et al., 2018, [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)), a reinforcement learning algorithm, using PyTorch and Python 3.

This report covers the following areas:

* The Learning Algorithm: Twined Delayed DDPG (TD3)
* The Results
* Ideas for Future Work

## The Learning Algorithm: Twined Delayed DDPG (TD3)

The Twined Delayed DDPG (TD3) algorithm aims to address function approximation errors, which lead to overestimating value estimates in value-based, actor-critic, deep Q-learning methods. The TD3 algorithm achieves this building on the Deep Deterministic Policy Gradient algorithm (DDPG) (Lillicrap et al., 2015) by taking the minimum value between a pair of critic networks to limit overestimation and by delaying policy updates to reduce per-update error which further improves performance.

In their paper, [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018), the authors evaluate their modethod on a suite of OpenAI gym tasks. At the time of publication, the algorithm outperformed the state of art on every environment tested.

Twined Delayed DDPG (TD3) focuses on continuous control environments. The authors argue that the overestimation of value estimates is well studied in problems with discrete action spaces but that this issue had been ignored in the continuous control domain.

### Implementation notes

The implementation I chose is based on the TD3 implementation in [Modularized Implementation of Deep RL Algorithms in PyTorch](https://github.com/ShangtongZhang/DeepRL). This is a codebase base where functionality such as the replay buffer is well abstracted and the implementation of each algorithm is easy to read.

I modified this implementation of TD3 to support environments that contain multiple instances of the same environment. This can reduce the number back and forths between the environment and python code and can improve wall clock training speeds.

I tested my changes on [Marathon Environments](https://arxiv.org/abs/1902.09097) (Booth et al., 2019), which is a suite of continuous control benchmarks that I created using Unity. Marathon Environments includes some basic environments simular to Mujoco, such as Hopper, Walker, Ant, and Humanoid, along with some more complex versions which have to learn to navigate a randomize terrain as well as motion matching (simular to DeepMimic) where the agent must match motion captured data.

I found I was able to train many of the Marathon Environments. However, I found that the TD3 environment became less stable with more environment instances. This was helped by increasing the number of warm up steps to 100k.

When I switched to the Tennis environment, I found it was able to train with no further tweaks to the hyperparameters.

### Hyperparameters

* ```num_workers = 20``` - this is the number of concurrent environments.
* ```config.mini_batch_size = 2000``` - this is the batch size. I used a formula of ```100 * num_workers``` and found this stabilized learning well.
* ```warm_up = int(1e5)``` - number of random actions before training starts
* ```max_steps = int(3e6)``` = the max number of training steps to take
* ```memory_size = int(1e6)``` = replay buffer size
* ```discount = 0.99``` = discount rate
* ```random_process_fn = GaussianProcess(std=LinearSchedule(0.1)``` = action noise
* ```td3_noise = 0.2``` = next_action noise
* ```td3_noise_clip = 0.5``` = next_action noise clip rate
* ```td3_delay = 2``` = Number of training steps between critic updates
* ```target_network_mix = 5e-3``` = soft update value

### Neural Network Model Architecture

Both the actor and critic networks use four network layers, an input layer, two hidden layers, and an output layer. The layers have the following size and activation units.

```
input layer = 33 (number of observations) with relu activation
hidden layer = 32 with relu activation
hidden layer 2 = 32 with relu activation
output layer = 4 (number of actions)
```

## Results

I was able to achieve an average score of 30+ over 100 episodes after 115 episodes.

![episode_returns_ave100]

The raw output of the average score over the 20 environments at the end of each episode looks like this:

![raw_episode_returns]

Here is a video of the successful agent:

![Tennis Environment][p2_solution]

I also check two additional runs to ensure that the results are robust and reproducible. All three runs resulted in simular scores:

![average_over_3_runs]

## Ideas for Future Work

Future work might want to explore a number of different directions:

### Implement a PD controller

Many environments in the continuous control domain benefit from using a PD controller. The paper, [Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?](https://www.cs.ubc.ca/~van/papers/2017-SCA-action/2017-SCA-action.pdf) (Peng et al.,2017) explores the value of using PD controllers. The idea I propose exploring is to implement the PD controller on top of the Tennis environment.

### Explore Action Smoothing

In [DReCon: Data-Driven Responsive Control of Physics-Based Characters](https://t.co/Ahn4UiM5EI?amp=1) (BERGAMIN et al., 2019)

A recursive exponentially weighted moving average filter is used, yt = βat +(1−β)yt−1, where yt and at are the output and input of the filter at time t respectively. Here, the action stiffness β is a hyperparameter controlling the filter strength

### Other State of Art Continuous Control algorithms

The [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) algorithm has been shown to give SOA results and might perform faster on this environment due to its focus on maximizing entropy.
