[average_over_3_runs]: images/average_over_3_runs.png "average_over_3_runs"
[episode_returns_ave100]: images/episode_returns_ave100.png "episode_returns_ave100"
[p3_solution]: images/p3_solution.gif "p2_solution"
[p3_solution_3m_steps]: images/p3_solution_3m_steps.gif "p3_solution_3m_steps"
[p3_solution_50k_steps]: images/p3_solution_50k_steps.gif "p3_solution_50k_steps"
[raw_average_over_3_runs]: images/raw_average_over_3_runs.png "raw_average_over_3_runs"
[raw_episode_returns]: images/raw_episode_returns.png "raw_episode_returns"
[trained model]: images/trained_model.png "trained model"

# Udacity Reinforcement Learning Nanodagree - Project Three

Joe Booth May 2019

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

I modified this implementation of TD3 to support environments that contain multiple agents and self-play.

I tested my changes on [Marathon Environments](https://arxiv.org/abs/1902.09097) (Booth et al., 2019), which is a suite of continuous control benchmarks that I created using Unity. Marathon Environments includes some basic environments simular to Mujoco, such as Hopper, Walker, Ant, and Humanoid, along with some more complex versions which have to learn to navigate a randomize terrain as well as motion matching (simular to DeepMimic) where the agent must match motion captured data.

I found I was able to train many of the Marathon Environments. However, I found that the TD3 environment became less stable with more environment instances. This was helped by increasing the number of warm up steps to 100k.

When I switched to the Tennis environment, I found it was able to train with no further tweaks to the hyperparameters.

### Hyperparameters

* ```num_workers = 2``` - this is the number of agents.
* ```config.mini_batch_size = 2000``` - this is the batch size. I used a formula of ```1000 * num_workers``` and found this stabilized learning well.
* ```warm_up = int(1e4)``` - number of random actions before training starts
* ```max_steps = int(3e6)``` = the max number of training steps to take
* ```memory_size = int(1e5)``` = replay buffer size
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

I was able to achieve an average score of .5+ over 100 episodes after 1,2870 episodes.

![episode_returns_ave100]

Here is a video of the successful agent:

![Tennis Environment][p3_solution]

Here is the agent after 50k steps. We see it plays well, however, the right and agent has an edge.

![50k steps][p3_solution_50k_steps]

Here is the agent after 3m steps. We see both agents now play at the same level.

![3m steps][p3_solution_3m_steps]


## Ideas for Future Work

Future work might want to explore training with less steps. For example the [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) algorithm has been shown to give SOA results and might perform faster on this environment due to its focus on maximizing entropy.

