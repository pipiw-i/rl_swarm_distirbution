# -*- coding: utf-8 -*-
# @Time : 2022/5/11 下午8:34
# @Author :  wangshulei
# @FileName: exp.py
# @Software: PyCharm
import pandas as pd
import numpy as np


class Exp:
    def __init__(self,
                 exp_size,
                 batch_size,
                 obs_dim,
                 action_dim,
                 r_dim,
                 done_dim):
        self.exp_size = exp_size
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.r_dim = r_dim
        self.done_dim = done_dim
        self.exp_length = obs_dim * 2 + action_dim + r_dim + done_dim
        self.exp_pool = pd.DataFrame(np.zeros([self.exp_size, self.exp_length]))  # 建立经验池
        self.exp_index = 0
        self.net_can_learn = False
        self.exp_full = False

    def exp_store(self, s, a, r, s_, done):
        """
        存储经验
        :param s: 状态
        :param a: 动作
        :param r: 回报
        :param s_: 下一个状态
        :param done: 是否完成游戏
        :return:
        """
        experience = []
        for i in range(self.exp_length):
            if i < self.obs_dim:
                experience.append(s[i])
            elif self.obs_dim <= i < self.obs_dim + self.action_dim:
                experience.append(a[i - self.obs_dim])
            elif self.obs_dim + 1 <= i < self.obs_dim + self.action_dim + 1:
                experience.append(r)
            elif self.obs_dim + 2 <= i < self.obs_dim * 2 + self.action_dim + 1:
                experience.append(s_[i - self.obs_dim - self.action_dim - 1])
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

    def can_learn(self):
        return self.net_can_learn

    def is_full(self):
        return self.exp_full

    def get_all_exp(self):
        return self.exp_pool

    def get_now_index(self):
        return self.exp_index

    def sample(self):
        if not self.can_learn():
            print("还没有到一个batch的数据")
            raise NotImplementedError
        else:
            if self.exp_full:
                exp = self.exp_pool.sample(self.batch_size)
                index = exp.index
            else:
                exp = self.exp_pool.loc[:self.exp_index - 1, :].sample(self.batch_size)
                index = exp.index
            return index, exp
