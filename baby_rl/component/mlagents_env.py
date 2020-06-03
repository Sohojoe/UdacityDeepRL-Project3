from unityagents import UnityEnvironment
import numpy as np
from unityagents import BrainInfo
import logging
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import gym
from gym import error, spaces
import os
from sys import platform

DEFAULT_EDITOR_PORT = 5004

GymMultiStepResult = Tuple[List[np.ndarray], List[float], List[bool], Dict]

class MLAgentsEnv(gym.Env):
    def __init__(
        self,
        environment_name: str,
        num_spawn_envs: int = 1,
        worker_id: int = 0,
        marathon_envs_path: str = None,
        no_graphics: bool = False,
        use_editor: bool = False,
        inference: bool = False,
    ):    
        base_port = 5005
        # use if we want to work with Unity Editoe
        if use_editor:
            base_port = DEFAULT_EDITOR_PORT
            marathon_envs_path = None
        elif marathon_envs_path is None:
            marathon_envs_path = 'Tennis'
            if platform == "win32":
                marathon_envs_path = os.path.join('Tennis_Windows_x86_64', 'Tennis.exe')
            base_port += worker_id
            worker_id = 0

        self._env = UnityEnvironment(
            marathon_envs_path,
            worker_id = worker_id,
            base_port=base_port,
            no_graphics=no_graphics)

        self._previous_step_result: BrainInfo = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False

        self.brain_name = self._env.brain_names[0]
        self.name = self.brain_name
        # self.group_spec = self._env.get_agent_group_spec(self.brain_name)
        self._brain = self._env.brains[self.brain_name]
        self._train_mode = not inference
        env_info = self._env.reset(train_mode=self._train_mode)[self.brain_name]
        self._n_agents = len(env_info.agents)

        print('Number of agents:', self._n_agents)

        # size of each action
        self._action_size = self._brain.vector_action_space_size
        print('Size of each action:', self._action_size)
        high = np.array([1] * self._action_size)
        self._action_space = spaces.Box(-high, high, dtype=np.float32)

        # examine the state space 
        self._states = env_info.vector_observations
        self._state_size = self._states.shape[1]
        high = np.array([np.inf] * self._state_size)
        self._observation_space = spaces.Box(-high, high, dtype=np.float32)

        print('There are {} agents. Each observes a state with length: {}'.format(self._states.shape[0], self._state_size))
        print('The state for the first agent looks like:', self._states[0])

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        env_info = self._env.reset(train_mode=self._train_mode)[self.brain_name] 
        self.game_over = False
        states = env_info.vector_observations 
        return states

    def step(self, action: List[Any]) -> GymMultiStepResult:
        if isinstance(action, list):
            action = np.array(action)
        action = np.array(action).reshape((self._n_agents, self._action_size))
        # action = self._sanitize_action(action)
        env_info = self._env.step(action)[self.brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        dones = [1 if t or self._env.global_done else 0 for t in env_info.local_done]
        dones = np.array(env_info.local_done)
        result = (
            next_states, 
            rewards, 
            dones,
            {"batched_step_result": env_info}
        )
        if self._env.global_done:
            print ('global_done')
        return result

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        # logger.warning("Could not seed environment %s", self.name)
        return

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents