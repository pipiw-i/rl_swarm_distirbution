# -*- coding: utf-8 -*-
# @Time : 2022/5/20 下午4:57
# @Author :  wangshulei
# @FileName: convert_obs.py
# @Software: PyCharm
from RL_algorithm_package.rddpg.policy.lstm import Lstm
import tensorflow as tf


class convert:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.lstm_model = Lstm(obs_dim)

    def convert_obs(self, obs):
        obs = tf.expand_dims(obs, axis=0)
        obs_out = self.lstm_model.lstm_model(obs)
        return obs_out * 50  # 使输出到大概分布在-2到2，后续容易更新


