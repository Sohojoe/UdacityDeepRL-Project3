from marathon_envs.envs import MarathonEnvs
import pathlib
import os
import numpy as np


def test_gym(name):
    num_envs = 1
    num_envs = 3
    num_envs = 64
    id_num = 0
    env = MarathonEnvs(name, num_envs)
    obs = env.reset()
    while True:
        actions = [env.action_space.sample() for _ in range(num_envs)]
        obs, rewards, dones, info = env.step(actions)
        env.step(actions)

if __name__ == '__main__':
    # game = 'Hopper-v0'
    game = 'Walker2d-v0'
    # game = 'Ant-v0'
    # game = 'MarathonMan-v0'
    test_gym(game)
