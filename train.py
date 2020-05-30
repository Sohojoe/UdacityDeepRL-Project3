from baby_rl import *

# TD3
def td3_continuous(**kwargs):
    # set up the default config and add the args
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    # set number of workers and batch size
    config.num_workers = 20
    config.mini_batch_size = 2000
    config.num_mini_batch = 1
    # number of random actions before training starts
    config.warm_up = int(1e5)
    # set the max number of taining steps to take
    config.max_steps = int(3e6)

    # set up the environment
    config.task_fn = lambda: config.eval_env
    config.eval_env = Task(config.game, 20)
    # 
    config.eval_interval = int(1e5)
    config.eval_episodes = 3
    # set how often to save the model
    config.save_interval = int(1e5)

    # set the nn size and learning weights
    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (32, 32), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (32, 32), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    # create the replay buffer and hyper parameters
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=config.mini_batch_size)
    config.discount = 0.99
    # set the random procress for actions
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    # how much noise to add
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    # delay between action steps and soft updates
    config.td3_delay = 2
    # soft update value
    config.target_network_mix = 5e-3

    # start training
    run_steps(TD3Agent(config))

if __name__ == '__main__':
    # create sub directories
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    # set one thread as this seams to train faster
    set_one_thread()
    # set the seed
    random_seed()
    # choose between cpu(-1) and gpu(0)
    select_device(-1)
    # select_device(0)

    # train the Tennis environment
    game = 'Tennis'
    td3_continuous(game=game)