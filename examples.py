#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from baby_rl import *


# TD3
def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e5)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    game = 'CartPole-v0'
    # dqn_feature(game=game)
    # quantile_regression_dqn_feature(game=game)
    # categorical_dqn_feature(game=game)
    # a2c_feature(game=game)
    # n_step_dqn_feature(game=game)
    # option_critic_feature(game=game)

    # game = 'MountainCarContinuous-v0'
    game = 'Pendulum-v0'
    # game = 'HalfCheetah-v2'
    # game = 'Hopper-v2'
    # a2c_continuous(game=game)
    # ppo_continuous(game=game)
    # ddpg_continuous(game=game)
    td3_continuous(game=game)

    game = 'BreakoutNoFrameskip-v4'
    # dqn_pixel(game=game)
    # quantile_regression_dqn_pixel(game=game)
    # categorical_dqn_pixel(game=game)
    # a2c_pixel(game=game)
    # n_step_dqn_pixel(game=game)
    # option_critic_pixel(game=game)