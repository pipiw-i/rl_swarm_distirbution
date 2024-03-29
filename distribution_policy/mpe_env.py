# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午4:12
# @Author :  wangshulei
# @FileName: mpe_env.py
# @Software: PyCharm
"""
环境仿真
"""
import mpe.scenarios as scenarios
from mpe.environment_distribution import MultiAgentEnv
from script.utils import space_n_to_shape_n


class mpe_env:
    def __init__(self,
                 mpe_env_name,
                 seed,
                 test=False):
        self.mpe_env_name = mpe_env_name
        self.seed = seed
        self.test = test
        self.mpe_env = self.env_init()
        self.world = None
        self.scenario = None

    def env_init(self):
        if self.test:
            self.scenario = scenarios.load(self.mpe_env_name + '_test.py').Scenario()
        else:
            self.scenario = scenarios.load(self.mpe_env_name + '.py').Scenario()
        # create world
        self.world = self.scenario.make_world()

        # create multiagent environment
        env = MultiAgentEnv(self.world, self.scenario.reset_world, self.scenario.reward, self.scenario.observation,
                            info_callback=None, benchmark_callback=self.scenario.get_obs_landmarks_pos,
                            get_attack_number_callback=self.scenario.get_attack_number,
                            shared_viewer=True)
        env.seed(self.seed)
        return env

    def get_attack_index(self, n_game):
        return self.scenario.get_attack_agent_index(self.world, n_game)

    def get_action_space(self):
        act_shape_n = space_n_to_shape_n(self.mpe_env.action_space)
        return act_shape_n[0][0]

    def set_move(self):
        self.scenario.set_move()

    def get_obs_space(self):
        obs_shape_n = space_n_to_shape_n(self.mpe_env.observation_space)
        return obs_shape_n[0][0]

    def get_agent_number(self):
        return self.mpe_env.n
