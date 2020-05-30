#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class TD3Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.training_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            # lazy initialize the states (via resetting the environment)
            self.random_process.reset_states()
            self.state = self.task.reset()
            # use a normalizer for the states
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            # select a randome action if still in warm up phase
            action = [self.task.action_space.sample() for _ in range(config.num_workers)]
        else:
            # sample an action using states
            action = self.network(self.state)
            action = to_np(action)
            # add small random noise
            action += self.random_process.sample()
        # clip actions between low and high
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        # step using the action
        next_state, reward, done, info = self.task.step(action)
        # normalize next_state
        next_state = self.config.state_normalizer(next_state)
        # log output
        self.record_online_return(info)
        # normalize reward
        reward = self.config.reward_normalizer(reward)

        # add experiances to replay buffer
        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)

        # handle done
        if done[0]:
            self.random_process.reset_states()
        # 
        self.state = next_state
        self.total_steps += config.num_workers
        self.training_steps += 1

        # update the nn if warm up complete
        if self.replay.size() >= config.warm_up:
            for _ in range(config.num_mini_batch):
                # sample experiances
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = tensor(states)
                actions = tensor(actions)
                rewards = tensor(rewards).unsqueeze(-1)
                next_states = tensor(next_states)
                mask = tensor(1 - terminals).unsqueeze(-1)

                # get current next action and add noise
                a_next = self.target_network(next_states)
                noise = torch.randn_like(a_next).mul(config.td3_noise)
                noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

                min_a = float(self.task.action_space.low[0])
                max_a = float(self.task.action_space.high[0])
                a_next = (a_next + noise).clamp(min_a, max_a)

                # get q based on next state and next action
                q_1, q_2 = self.target_network.q(next_states, a_next)
                target = rewards + config.discount * mask * torch.min(q_1, q_2)
                target = target.detach()

                # calculate loss
                q_1, q_2 = self.network.q(states, actions)
                critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

                # preform backprop
                self.network.zero_grad()
                critic_loss.backward()
                self.network.critic_opt.step()


            if self.training_steps % config.td3_delay:
                # update the policy network
                action = self.network(states)
                policy_loss = -self.network.q(states, action)[0].mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                # perform soft update
                self.soft_update(self.target_network, self.network)