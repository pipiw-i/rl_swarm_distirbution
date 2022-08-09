# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午4:12
# @Author :  wangshulei
# @FileName: mpe_env.py
# @Software: PyCharm
"""
多目标环境
"""
import mpe.scenarios as scenarios
from mul_target_test.environment_mul_target import MultiAgentEnv
from script.utils import space_n_to_shape_n


class mpe_env:
    def __init__(self,
                 mpe_env_name,
                 seed,
                 number_agent,
                 number_landmark,
                 need_search_agent,
                 grouping_ratio):
        self.mpe_env_name = mpe_env_name
        self.seed = seed
        self.need_search_agent = need_search_agent
        self.number_agent = number_agent
        self.number_landmark = number_landmark
        self.grouping_ratio = grouping_ratio  # 分组比例
        self.world = None
        self.scenario = None
        self.mpe_env = self.env_init()

    def env_init(self):
        self.scenario = scenarios.load(self.mpe_env_name).Scenario()
        self.set_agent_landmark_numbers(self.number_agent, self.number_landmark, self.grouping_ratio)
        # create world
        self.world = self.scenario.make_world()

        # create multiagent environment
        env = MultiAgentEnv(self.world, self.need_search_agent, self.scenario.reset_world,
                            reward_callback=None,
                            observation_callback=self.scenario.test_observation,
                            info_callback=None, land_callback=self.scenario.get_obs_landmarks_pos,
                            get_attack_number_callback=self.scenario.get_attack_number,
                            shared_viewer=True)
        env.seed(self.seed)
        return env

    def get_attack_index(self, n_game):
        return self.scenario.get_attack_agent_index(self.world, n_game)

    def set_agent_landmark_numbers(self, agent_numbers, landmark_numbers, grouping_ratio):
        # 设置无人机以及目标的数目，该步要在make world之前进行
        self.scenario.set_agent_landmark_numbers(agent_numbers, landmark_numbers, grouping_ratio)

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
