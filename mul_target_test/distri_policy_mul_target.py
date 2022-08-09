# -*- coding: utf-8 -*-
# @Time : 2022/6/9 上午11:47
# @Author :  wangshulei
# @FileName: distri_policy.py
# @Software: PyCharm
"""
多目标策略
"""
import tensorflow as tf

from distribution_policy.distri_actor_critic_net import test_actor as actor


class maddpg_policy:
    def __init__(self,
                 agent_index,
                 obs_dim=7,
                 action_dim=5,
                 action_span=0.5,
                 ):
        self.agent_index = agent_index
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_span = action_span
        self.actor_pred = actor()

    def get_action(self, obs):
        return self.__choose_test_action(obs)

    def __choose_test_action(self, s):
        s = tf.reshape(s, [1, s.shape[0]])
        a_hit, a_move = self.actor_pred.actor(s)
        a = tf.concat([a_hit, a_move], axis=-1)
        return a[0]

    def load_test_models(self, save_file, episode):
        self.actor_pred.actor = tf.keras.models.load_model(save_file + f'/maddpg_model/actor_pred/agent_0_{episode}.h5')
