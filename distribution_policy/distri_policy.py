# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午1:47
# @Author :  wangshulei
# @FileName: distri_policy.py
# @Software: PyCharm
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from distribution_policy.distri_actor_critic_net import actor
from distribution_policy.distri_actor_critic_net import critic


class maddpg_policy:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 agent_number,
                 actor_learning_rate,
                 critic_learning_rate,
                 action_span,
                 soft_tau,
                 log_dir,
                 gamma=0.95,
                 actor_name='actor',
                 critic_name='critic',
                 ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_span = action_span
        self.agent_number = agent_number
        self.soft_tau = soft_tau
        self.gamma = gamma
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir + '/maddpg_logs')
        self.actor_pred_list = []
        self.actor_target_list = []
        self.critic_pred_list = []
        self.critic_target_list = []
        for agent_index in range(agent_number):
            actor_pred = actor(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                               actor_learning_rate=actor_learning_rate, agent_index=agent_index,
                               trainable=True, actor_name=actor_name + 'pred',
                               action_span=action_span
                               )
            actor_target = actor(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                                 actor_learning_rate=actor_learning_rate, agent_index=agent_index,
                                 trainable=False, actor_name=actor_name + 'target',
                                 action_span=action_span
                                 )
            critic_pred = critic(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                                 critic_learning_rate=critic_learning_rate, agent_index=agent_index,
                                 trainable=True, critic_name=critic_name + 'pred',
                                 )
            critic_target = critic(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                                   critic_learning_rate=critic_learning_rate, agent_index=agent_index,
                                   trainable=False, critic_name=critic_name + 'target',
                                   )
            self.actor_pred_list.append(actor_pred)
            self.actor_target_list.append(actor_target)
            self.critic_pred_list.append(critic_pred)
            self.critic_target_list.append(critic_target)

    def __choose_action(self, s, agent_index, explore_range):
        s = tf.reshape(s, [1, s.shape[0]])
        a_hit, a_move = self.actor_pred_list[agent_index].actor(s)
        # random_hit = np.random.uniform(0, 1, 1)[0]
        # if random_hit < explore_range:
        #     a_hit = [np.random.choice([0, 1], 1)]
        # else:
        #     a_hit = a_hit
        # add noise
        u_h = tfp.distributions.Normal(a_hit, explore_range)
        action_h = tf.squeeze(u_h.sample(1), axis=0)[0]
        action_h = tf.clip_by_value(action_h, clip_value_min=0, clip_value_max=1)
        # 正式的测试中，可以去掉这个噪声 直接返回动作a[0]即可，这里是训练用，所以要加入噪声，使其能够充分的探索环境
        u = tfp.distributions.Normal(a_move, explore_range)
        action = tf.squeeze(u.sample(1), axis=0)[0]
        action = tf.clip_by_value(action, clip_value_min=-self.action_span, clip_value_max=self.action_span)
        act = tf.concat([action_h, action], axis=-1)
        return act

    def get_all_action(self, obs_n, explore_range):
        action_n = [self.__choose_action(obs, agent_index, explore_range) for obs, agent_index in
                    zip(obs_n, range(self.agent_number))]
        action_n = [action.numpy() for action in action_n]
        return action_n

    def get_all_test_action(self, obs_n, attack_number_index_list, show=True):
        action_n = [self.__choose_test_action(obs, agent_index) for obs, agent_index in
                    zip(obs_n, range(self.agent_number))]
        action_real = []
        if not attack_number_index_list and show:
            for index, action in enumerate(action_n):
                action_real.append(np.array([0, 0, 0, 0, 0]))
        elif attack_number_index_list and show:
            for index, action in enumerate(action_n):
                if index in attack_number_index_list:
                    # action_real.append(np.array([1, 0, 0, 0, 0]))
                    action_real.append(action.numpy())
                else:
                    action_real.append(np.array([0, 0, 0, 0, 0]))
        else:
            for index, action in enumerate(action_n):
                action_real.append(action.numpy())
        return action_real

    def __choose_test_action(self, s, agent_index):
        s = tf.reshape(s, [1, s.shape[0]])
        a_hit, a_move = self.actor_pred_list[agent_index].actor(s)
        a = tf.concat([a_hit, a_move], axis=-1)
        return a[0]

    def synchronization_parameters(self):
        parameters = []
        for model in self.actor_pred_list:
            parameters.append(model.actor.get_weights())
        param = []
        for i in range(len(parameters[0])):
            param_1 = np.add(parameters[0][i] * 0.25, parameters[1][i] * 0.25)
            param_2 = np.add(parameters[2][i] * 0.25, parameters[3][i] * 0.25)
            param.append(np.add(param_1, param_2))
        for model in self.actor_pred_list:
            model.actor.set_weights(param)

        parameters = []
        for model in self.actor_target_list:
            parameters.append(model.actor.get_weights())
        param = []
        for i in range(len(parameters[0])):
            param_1 = np.add(parameters[0][i] * 0.25, parameters[1][i] * 0.25)
            param_2 = np.add(parameters[2][i] * 0.25, parameters[3][i] * 0.25)
            param.append(np.add(param_1, param_2))
        for model in self.actor_target_list:
            model.actor.set_weights(param)

        parameters = []
        for model in self.critic_pred_list:
            parameters.append(model.critic.get_weights())
        param = []
        for i in range(len(parameters[0])):
            param_1 = np.add(parameters[0][i] * 0.25, parameters[1][i] * 0.25)
            param_2 = np.add(parameters[2][i] * 0.25, parameters[3][i] * 0.25)
            param.append(np.add(param_1, param_2))
        for model in self.critic_pred_list:
            model.critic.set_weights(param)

        parameters = []
        for model in self.critic_target_list:
            parameters.append(model.critic.get_weights())
        param = []
        for i in range(len(parameters[0])):
            param_1 = np.add(parameters[0][i] * 0.25, parameters[1][i] * 0.25)
            param_2 = np.add(parameters[2][i] * 0.25, parameters[3][i] * 0.25)
            param.append(np.add(param_1, param_2))
        for model in self.critic_target_list:
            model.critic.set_weights(param)

    def soft_param_update(self, target_model, pred_model):
        """
        采用软更新的方式进行参数的更新，不采用DQN中的直接赋值操作，也可以采用别的软更新方式来实现。
        :param pred_model: 预测网络
        :param target_model: 目标网络
        """
        param_target = target_model.get_weights()
        param_pred = pred_model.get_weights()
        param = []
        for i in range(len(param_target)):
            param_target[i] = param_target[i] * (1 - self.soft_tau)
            param_pred[i] = param_pred[i] * self.soft_tau
            param.append(np.add(param_pred[i], param_target[i]))
        target_model.set_weights(param)

    def update(self, all_agent_exp, exp_index_n):
        exp_n = []
        for agent_index, exp_index in zip(range(self.agent_number), exp_index_n):
            now_agent_index, obs, action, rew, obs_, done = all_agent_exp[agent_index].get_exp_from_index(exp_index)
            exp = [now_agent_index, obs, action, rew, obs_, done]
            exp_n.append(exp)
        # 更新网络
        with tf.GradientTape(persistent=True) as Tape:
            for agent_index in range(self.agent_number):
                obs = exp_n[agent_index][1]  # 得到当前的状态obs --> exp_n[agent_index][1]
                a_hit, a_move = self.actor_pred_list[agent_index].actor(obs)  # 得到当前agent关于自己的obs的动作值
                action = tf.concat([a_hit, a_move], axis=-1)
                if agent_index == 0:
                    all_obs = tf.convert_to_tensor(obs)
                    all_action = action
                else:
                    all_obs = tf.concat([all_obs, obs], axis=-1)
                    all_action = tf.concat([all_action, action], axis=-1)
            # 得到所有的actor_target网络关于自身obs_的值，更新critic网络
            for agent_index in range(self.agent_number):
                obs_ = exp_n[agent_index][4]  # 得到当前的状态obs --> exp_n[agent_index][1]
                a_hit_, a_move_ = self.actor_target_list[agent_index].actor(obs_)  # 得到当前agent关于自己的obs的动作值
                action_ = tf.concat([a_hit_, a_move_], axis=-1)
                if agent_index == 0:
                    all_obs_ = tf.convert_to_tensor(obs_)
                    all_action_ = action_
                else:
                    all_obs_ = tf.concat([all_obs_, obs_], axis=-1)
                    all_action_ = tf.concat([all_action_, action_], axis=-1)
            # 对agent依次进行更新
            for agent_index in range(self.agent_number):
                actor_pred = self.actor_pred_list[agent_index].actor
                critic_pred = self.critic_pred_list[agent_index].critic
                critic_target = self.critic_target_list[agent_index].critic
                # 得到当前更新agent的所有经验
                action = exp_n[agent_index][2]
                reward = exp_n[agent_index][3]
                # 更新actor,每一个智能体的actor需要他本身的critic_pred，输入状态动作，然后最大化这个值
                Q_pred = critic_pred([all_obs, all_action])
                actor_pred_loss = - tf.math.reduce_mean(Q_pred)
                gradients = Tape.gradient(actor_pred_loss, actor_pred.trainable_variables)
                actor_pred.optimizer.apply_gradients(zip(gradients, actor_pred.trainable_variables))
                # 更新critic网络
                Q_pred_critic = critic_pred([all_obs, action])
                Q_target_critic = reward + self.gamma * critic_target([all_obs_, all_action_])
                loss_critic = tf.keras.losses.mse(Q_target_critic, Q_pred_critic)
                loss_critic = tf.reduce_mean(loss_critic)
                critic_gradients = Tape.gradient(loss_critic, critic_pred.trainable_variables)
                critic_pred.optimizer.apply_gradients(zip(critic_gradients, critic_pred.trainable_variables))

        for agent_index in range(self.agent_number):
            self.soft_param_update(self.critic_target_list[agent_index].critic,
                                   self.critic_pred_list[agent_index].critic)
            self.soft_param_update(self.actor_target_list[agent_index].actor,
                                   self.actor_pred_list[agent_index].actor)

    def logs(self, score, learn_step):
        # print('... saving logs ...')
        with self.writer.as_default():
            tf.summary.scalar("score", score, learn_step)

    def save_models(self, save_file, episode):
        print('... saving models ...')
        if not os.path.exists(save_file + '/maddpg_model'):
            os.makedirs(save_file + '/maddpg_model')
            os.makedirs(save_file + '/maddpg_model/actor_pred')
            os.makedirs(save_file + '/maddpg_model/actor_target')
            os.makedirs(save_file + '/maddpg_model/critic_pred')
            os.makedirs(save_file + '/maddpg_model/critic_target')
        for agent_index in range(self.agent_number):
            self.actor_pred_list[agent_index].actor.save(
                save_file + f'/maddpg_model/actor_pred/agent_{agent_index}_{episode}.h5')
            self.actor_target_list[agent_index].actor.save(
                save_file + f'/maddpg_model/actor_target/agent_{agent_index}_{episode}.h5')
            self.critic_pred_list[agent_index].critic.save(
                save_file + f'/maddpg_model/critic_pred/agent_{agent_index}_{episode}.h5')
            self.critic_target_list[agent_index].critic.save(
                save_file + f'/maddpg_model/critic_target/agent_{agent_index}_{episode}.h5')

    def load_models(self, save_file, episode):
        print('... loading models ...')
        for agent_index in range(self.agent_number):
            self.actor_pred_list[agent_index].actor = tf.keras.models.load_model(
                save_file + f'/maddpg_model/actor_pred/agent_{agent_index}_{episode}.h5')
            self.actor_target_list[agent_index].actor = tf.keras.models.load_model(
                save_file + f'/maddpg_model/actor_target/agent_{agent_index}_{episode}.h5')
            self.critic_pred_list[agent_index].critic = tf.keras.models.load_model(
                save_file + f'/maddpg_model/critic_pred/agent_{agent_index}_{episode}.h5')
            self.critic_target_list[agent_index].critic = tf.keras.models.load_model(
                save_file + f'/maddpg_model/critic_target/agent_{agent_index}_{episode}.h5')

    def load_test_models(self, save_file, episode):
        print('... loading test models ...')
        for agent_index in range(self.agent_number):
            self.actor_pred_list[agent_index].actor = tf.keras.models.load_model(
                save_file + f'/maddpg_model/actor_pred/agent_0_{episode}.h5')
