# -*- coding: utf-8 -*-
# @Time : 2022/5/20 上午10:43
# @Author :  wangshulei
# @FileName: env_test.py
# @Software: PyCharm
import numpy as np

import mpe.scenarios as scenarios
from mpe.environment_distribution_s_reward import MultiAgentEnv as MultiAgentEnv_s
from script.utils import space_n_to_shape_n


class mpe_env:
    def __init__(self,
                 mpe_env_name,
                 seed):
        self.mpe_env_name = mpe_env_name
        self.seed = seed
        self.mpe_env = self.env_init()

    def env_init(self):
        scenario = scenarios.load(self.mpe_env_name + '.py').Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv_s(world, scenario.reset_world, scenario.reward, scenario.observation,
                              benchmark_callback=scenario.get_obs_landmarks_pos,
                              info_callback=None,
                              shared_viewer=True)
        env.seed(self.seed)
        return env

    def get_space(self):
        obs_shape_n = space_n_to_shape_n(self.mpe_env.observation_space)
        act_shape_n = space_n_to_shape_n(self.mpe_env.action_space)
        return obs_shape_n[0][0], act_shape_n[0][0]

    def get_agent_number(self):
        return self.mpe_env.n


obs_dim = 8
if __name__ == "__main__":
    world = mpe_env('simple_distribution', 65535)
    while True:
        obs = world.mpe_env.reset()
        for n_games in range(10):
            world.mpe_env.render()
            action_n = [np.array([1, 0, 1, 1, 0]), np.array([0, 0, 1, 0, 1]),
                        np.array([1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 1])]
            new_obs_n, reward_n, done_n, info_n = world.mpe_env.step(action_n)
            print(f"reward_n is {reward_n}")
