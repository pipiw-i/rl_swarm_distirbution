# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午2:38
# @Author :  wangshulei
# @FileName: distri_exp.py
# @Software: PyCharm
import numpy as np
import pandas as pd

from distribution_policy.exp import Exp


class SharedExp(Exp):
    """
    一个agent的经验
    """

    def __init__(self, exp_size, batch_size, obs_dim, action_dim, r_dim, done_dim, agent_number, agent_index):
        super().__init__(exp_size, batch_size, obs_dim, action_dim, r_dim, done_dim)
        self.agent_number = agent_number
        self.action_exp_length = self.action_dim * agent_number
        self.exp_length = obs_dim * 2 + self.action_exp_length + r_dim + done_dim
        self.exp_pool = pd.DataFrame(np.zeros([self.exp_size, self.exp_length]))  # 建立经验池
        self.agent_index = agent_index
        self.exp_index = 0
        self.net_can_learn = False
        self.exp_full = False

    def __one_agent_exp_store(self, s, a_n, r, s_, done):
        """
        存储经验
        :param s: 状态
        :param a_n: 动作,是所有智能体的联合动作,因为critic网络需要输入所有的智能体动作来判断价值
        :param r: 回报
        :param s_: 下一个状态
        :param done: 是否完成游戏
        :return:
        """
        experience = []
        for i in range(self.exp_length):
            if i < self.obs_dim:
                experience.append(s[i])
            elif self.obs_dim <= i < self.obs_dim + self.action_exp_length:
                experience.append(a_n[i - self.obs_dim])
            elif self.obs_dim + 1 <= i < self.obs_dim + self.action_exp_length + 1:
                experience.append(r)
            elif self.obs_dim + 2 <= i < self.obs_dim * 2 + self.action_exp_length + 1:
                experience.append(s_[i - self.obs_dim - self.action_exp_length - 1])
            else:
                experience.append(done)
        self.exp_pool.loc[self.exp_index] = experience
        self.exp_index += 1
        # 判断能否开始训练，以及经验池是否已经满了
        if self.exp_index >= self.batch_size:
            self.net_can_learn = True
        if self.exp_index == self.exp_size:
            self.exp_full = True
            self.exp_index = 0

    def exp_store(self, s_n, a_n, r_n, s__n, done_n):
        all_agent_action = []
        for one_agent_action in a_n:
            for a in one_agent_action:
                all_agent_action.append(a)
        self.__one_agent_exp_store(s_n[self.agent_index], all_agent_action,
                                   r_n[self.agent_index], s__n[self.agent_index],
                                   done_n[self.agent_index])

    def get_exp_from_index(self, index):
        exp = self.exp_pool.loc[index]
        obs = exp.loc[:, :self.obs_dim - 1].to_numpy()
        action = exp.loc[:, self.obs_dim:self.obs_dim - 1 + self.action_exp_length].to_numpy()
        rew = np.array(exp.loc[:, self.obs_dim + self.action_exp_length]).reshape(self.batch_size, 1)
        obs_ = np.array(exp.loc[:, self.obs_dim + self.action_exp_length + 1:self.obs_dim * 2 + self.action_exp_length])
        done = np.array(exp.loc[:, self.obs_dim * 2 + self.action_exp_length + 1])
        return self.agent_index, obs, action, rew, obs_, done
