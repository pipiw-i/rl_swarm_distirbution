# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午1:45
# @Author :  wangshulei
# @FileName: distri_actor_critic_net.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


class critic:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 agent_number,
                 critic_learning_rate,
                 agent_index,
                 trainable,
                 critic_name, ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_number = agent_number
        self.critic_learning_rate = critic_learning_rate
        self.critic = self.__critic_net(trainable, agent_index, critic_name)

    def __critic_net(self, trainable, agent_index, critic_name):
        # MADDPG的评论家是根据全局的信息来进行评论的，所以这里需要知道其他所有智能体的观测值以及动作***
        input_magent_s = tf.keras.Input(shape=(self.obs_dim * self.agent_number,), dtype="float32")
        input_magent_a = tf.keras.Input(shape=(self.act_dim * self.agent_number,), dtype="float32")
        input_critic = tf.concat([input_magent_s, input_magent_a], axis=-1)
        dense1 = tf.keras.layers.Dense(128, activation='relu')(input_critic)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        critic_output = tf.keras.layers.Dense(1)(dense2)
        critic_model = tf.keras.Model(inputs=[input_magent_s, input_magent_a],
                                      outputs=critic_output,
                                      trainable=trainable,
                                      name=f'agent{agent_index}_{critic_name}')
        critic_model.compile(optimizer=tf.keras.optimizers.Adam(self.critic_learning_rate))
        return critic_model


class actor:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 agent_number,
                 actor_learning_rate,
                 agent_index,
                 trainable,
                 actor_name,
                 action_span):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_span = action_span
        self.agent_number = agent_number
        self.actor_learning_rate = actor_learning_rate
        self.actor = self.__actor_net(trainable, agent_index, actor_name, action_span)

    def __actor_net(self, trainable, agent_index, actor_name, action_span):
        # MADDPG的演员是根据自己智能体的观测值来得到动作的
        input_s = tf.keras.Input(shape=(self.obs_dim,), dtype="float32")
        dense1 = tf.keras.layers.Dense(128, activation='relu')(input_s)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        actor_output_4 = tf.keras.layers.Dense(self.act_dim - 1, activation='tanh')(dense2)  # 使用tanh输出动作
        actor_output_1 = tf.keras.layers.Dense(self.act_dim - 4, activation='sigmoid')(dense2)  # 使用sigmoid执行攻击分类
        actor_output = tf.keras.layers.Lambda(lambda x: x * np.array(action_span))(actor_output_4)
        actor_model = tf.keras.Model(inputs=input_s,
                                     outputs=[actor_output_1, actor_output],
                                     trainable=trainable,
                                     name=f'agent{agent_index}_{actor_name}')
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate))
        return actor_model


class test_actor:
    def __init__(self,
                 obs_dim=7,
                 act_dim=5,
                 actor_learning_rate=1e-3,
                 agent_index=0,
                 trainable=False,
                 action_span=0.5):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_span = action_span
        self.actor_learning_rate = actor_learning_rate
        self.actor = self.__actor_net(trainable, agent_index, action_span)

    def __actor_net(self, trainable, agent_index, action_span):
        # MADDPG的演员是根据自己智能体的观测值来得到动作的
        input_s = tf.keras.Input(shape=(self.obs_dim,), dtype="float32")
        dense1 = tf.keras.layers.Dense(128, activation='relu')(input_s)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        actor_output_4 = tf.keras.layers.Dense(self.act_dim - 1, activation='tanh')(dense2)  # 使用tanh输出动作
        actor_output_1 = tf.keras.layers.Dense(self.act_dim - 4, activation='sigmoid')(dense2)  # 使用sigmoid执行攻击分类
        actor_output = tf.keras.layers.Lambda(lambda x: x * np.array(action_span))(actor_output_4)
        actor_model = tf.keras.Model(inputs=input_s,
                                     outputs=[actor_output_1, actor_output],
                                     trainable=trainable,
                                     name=f'agent{agent_index}')
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate))
        return actor_model
