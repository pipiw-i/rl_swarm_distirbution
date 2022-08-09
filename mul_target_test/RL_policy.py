# -*- coding: utf-8 -*-
# @Time : 2022/7/10 下午13:21
# @Author :  wangshulei
# @FileName: RL_policy.py
# @Software: PyCharm
from mul_target_test.distri_policy_mul_target import maddpg_policy


class RL_policy:
    def __init__(self,
                 load_file,
                 agent_index,
                 n_test_times=1000):
        """
        :param load_file:  加载的rl策略文件位置
        :param n_test_times:  加载的rl策略的迭次数
        """
        self.load_file = load_file
        self.agent_index = agent_index
        self.obs_dim = 7
        self.act_dim = 5
        self.rl_model = self.RL_model(n_test_times)

    def RL_model(self, n_test_times):
        # policy初始化,单一智能体
        maddpg_agent = maddpg_policy(obs_dim=self.obs_dim,
                                     action_dim=self.act_dim, agent_index=self.agent_index)
        # 加载训练好的数据
        maddpg_agent.load_test_models(save_file=self.load_file,
                                      episode=n_test_times)
        return maddpg_agent

    def get_rl_action(self, obs):
        rl_action = self.rl_model.get_action(obs)
        return rl_action.numpy()
