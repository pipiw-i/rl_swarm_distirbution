# -*-coding:utf-8-*-
# @Time : 2022/7/9 下午22:36
# @Author :  wangshulei
# @FileName: total_policy.py
# @Software: PyCharm

import numpy as np
from mul_target_test.RL_policy import RL_policy
from mul_target_test.boids_policy import boids_policy
from mul_target_test.logs import Logs

"""
最终效果应该是每个智能体都有一份完全相同的策略，该策略包含rl策略，boids搜索策略，智能体可以根据通讯以及自身的信息，来做出决策
每个智能体的通讯内容：
1. 本身的攻击情况，True or False，如果是True，那么还要附带自己攻击目标，让其他的智能体可以区分
2. 智能体所能探测到的目标情况，目标位置
3. 智能体本身的位置信息
智能体接受到的信息：
智能体所能够接收到的信息，是完全基于通讯范围的，若不在范围内，获取不到他的通信内容。该通讯范围指的是直接通讯范围，间接通讯不考虑
1. 其他智能体的攻击情况
2. 其他智能体的位置
3. 其他智能体发送的目标位置信息
智能体需要的决策信息：
1. 其他智能体的攻击情况
2. 其他智能体的位置
3. 其他智能体发送的目标位置信息
决策流程：
1.每个智能体通讯，各个智能体互相接收能够接收的信息
2.首先得到其他智能体的目标位置信息
如果有n个，那么这个智能体自己的决策系统，根据次序，依次对目标进行决策。直到所有目标决策完毕。如果中间决定执行攻击，那么后续的目标不必进行
执行决策的流程如下：
(1) 执行第一个目标，执行完毕之后，重新接受一下周围的通讯内容，看看是否有多个无人机在执行当前目标，进行确认。
(2) 如果执行了攻击，那么将这个目标点隐去，该无人机的通讯内容也会失效。
(3) 继续执行别的目标
问题：在当前无人机的目标，并不一定是其他无人机的优先决策目标，这样可能会导致效果变差
如果没有，那么执行boids搜索
3.环境更新目标信息，更新无人机的动作，回到1循环进行。
"""


class AgentPolicy:
    def __init__(self,
                 agent_number,
                 landmark_number,
                 load_file='../distribution_policy/run_distribution_e3_syn_para_double_random_6_6_10',
                 n_test_times=6000,
                 boids_rule_1_distance=4.0,
                 boids_rule_2_distance=4.0,
                 boids_rule_3_distance=1.5,
                 grouping_ratio=0.5
                 ):
        self.agent_number = agent_number
        self.landmark_number = landmark_number
        self.load_file = load_file
        self.n_test_times = n_test_times
        self.grouping_ratio = grouping_ratio
        # 设计boids的参数
        self.boids_policy = boids_policy(self.agent_number, agent_com_size=4, max_vel=1,
                                         rule_1_distance=boids_rule_1_distance,
                                         rule_2_distance=boids_rule_2_distance,
                                         rule_3_distance=boids_rule_3_distance
                                         )
        # 设计强化学习策略参数
        self.rl_policy = RL_policy(load_file=self.load_file,
                                   agent_index=0,
                                   n_test_times=self.n_test_times)
        self.policy_logger = Logs(logger_name='policy_logs', log_file_name='policy_logs')

    def init_policy(self):
        """
        对每个智能体初始化自身的策略
        :return: [智能体1策略，智能体2策略，智能体3策略......]
        """
        agents_policy = []
        for agent_index in range(self.agent_number):
            agents_policy.append(self.one_agent_policy)
            # agents_policy.append(self.one_agent_policy_test_hover)
        self.policy_logger.add_logs("agent完全策略初始化成功...")
        return agents_policy

    def one_agent_policy_test_hover(self, obs, time_step, agent_index):
        """
        单一智能体的策略，该策略是一个函数，通过出入观测值，可以依据观测值输出动作
        :return: 单一智能体的策略函数
        """
        action = None
        agent, attack_number, agent.com_agent_index, all_pos, \
            all_relative_position, all_attack_agent_position, need_dist, all_vel, obs_landmarks = obs
        one_agent_boids_policy = self.boids_policy
        now_agent_pos = agent.state.p_pos
        now_agent_vel = agent.state.p_vel
        action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos, need_dist,
                                                                    now_agent_vel, all_vel, time_step)
        return action

    def one_agent_policy(self, obs, time_step, agent_index):
        """
        单一智能体的策略，该策略是一个函数，通过出入观测值，可以依据观测值输出动作
        :return: 单一智能体的策略函数
        """
        action = None
        # 该obs内的内容完全为部分观测内容。
        # agent 智能体本身的一些参数
        # attack_number 周围的攻击个数
        # agent.com_agent_index 能够通讯的智能体编号
        # all_pos 能够通讯的智能体位置
        # all_relative_position 强化学习观测值所用到的关联位置，用来计算平均位置（未攻击时，初始决策时）
        # all_attack_agent_position  强化学习观测值所用到的关联位置，用来计算平均位置（攻击时，迭代时）
        # need_dist boids需要的距离信息
        # all_vel boids需要的速度信息
        # obs_landmarks 当前目标所在的位置
        # obs_landmarks_feature 当前目标的标识
        agent, attack_number, agent.com_agent_index, all_pos, \
            all_relative_position, all_attack_agent_position, need_dist, all_vel, obs_landmarks, \
            obs_landmarks_feature, _ = obs
        # 这里首先对检测到的目标进行识别，如果不属于自己管辖的范围，则删除该目标，涉及到分组的问题
        new_targets = []  # 最终要执行的目标
        new_targets_feature = []
        for obs_landmark_index, obs_landmark in enumerate(obs_landmarks):
            obs_feature = obs_landmarks_feature[obs_landmark_index]
            # 首先判断agent_index的类别，即分组情况
            if agent_index >= int(self.grouping_ratio * self.agent_number):
                if obs_feature >= int(self.grouping_ratio * self.landmark_number):
                    new_targets.append(obs_landmark)
                    new_targets_feature.append(obs_feature)
            else:
                if obs_feature < int(self.grouping_ratio * self.landmark_number):
                    new_targets.append(obs_landmark)
                    new_targets_feature.append(obs_feature)
        # 智能体已经被毁坏了，即以及对目标发起过了攻击，那么使其追踪目标。
        if agent.is_destroyed:
            target = agent.attack_goal_pos
            now_pos = agent.state.p_pos
            act = 0.5 * (target - now_pos)
            action = np.array([0, act[0], 0, act[1], 0])
        # 未检测到目标，则采取boids编队
        elif len(new_targets) == 0:
            one_agent_boids_policy = self.boids_policy
            now_agent_pos = agent.state.p_pos
            now_agent_vel = agent.state.p_vel
            action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos, need_dist,
                                                                        now_agent_vel, all_vel, time_step)
        else:
            if attack_number == 0:
                # 只处理目标中的第一个,因为一次只能打击一个目标
                for obs_index, obs_landmark in enumerate(new_targets):
                    one_agent_rl_policy = self.rl_policy
                    entity_pos = []
                    if len(all_relative_position) == 0:
                        mean_other_pos = np.mean([agent.state.p_pos], axis=0)
                    else:
                        mean_other_pos = np.mean(all_relative_position, axis=0)
                    entity_pos.append(obs_landmark - agent.state.p_pos)
                    new_obs = np.concatenate(
                        [np.array([attack_number])] + [agent.state.p_pos] + [mean_other_pos] + entity_pos)
                    one_agent_boids_policy = self.boids_policy
                    now_agent_pos = agent.state.p_pos
                    now_agent_vel = agent.state.p_vel
                    # 先得到boids动作
                    action_boids = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos,
                                                                                      need_dist,
                                                                                      now_agent_vel, all_vel,
                                                                                      time_step)
                    # 依据强化学习的观测值，得到强化学习的动作，得出综合动作
                    action = [one_agent_rl_policy.get_rl_action(new_obs) + 2 * action_boids,
                              new_targets_feature[obs_index],
                              agent.index_number]
                    break
            else:
                for obs_index, obs_landmark in enumerate(new_targets):
                    if attack_number >= 4:
                        # 当执行攻击的无人机大于4时，即数目过多时
                        if not agent.attack:
                            # 如果当前的无人机不是攻击，执行boids
                            one_agent_boids_policy = self.boids_policy
                            now_agent_pos = agent.state.p_pos
                            now_agent_vel = agent.state.p_vel
                            action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos,
                                                                                        need_dist,
                                                                                        now_agent_vel, all_vel,
                                                                                        time_step)
                            break
                        else:
                            # 对于正在执行攻击的无人机采用强化学习策略
                            one_agent_boids_policy = self.boids_policy
                            now_agent_pos = agent.state.p_pos
                            now_agent_vel = agent.state.p_vel
                            action_boids = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos,
                                                                                              need_dist,
                                                                                              now_agent_vel, all_vel,
                                                                                              time_step)
                            # 只对攻击的无人机继续采用rl策略
                            one_agent_rl_policy = self.rl_policy
                            entity_pos = []
                            mean_other_pos = np.mean(all_attack_agent_position, axis=0)
                            entity_pos.append(obs_landmark - agent.state.p_pos)
                            # 由于进行了分组,所以在决策的时候，只与相同任务的智能体进行互动即可
                            new_obs = np.concatenate(
                                [np.array([attack_number])] + [agent.state.p_pos] + [mean_other_pos] + entity_pos)
                            action = [one_agent_rl_policy.get_rl_action(new_obs) + 2 * action_boids,
                                      new_targets_feature[obs_index],
                                      agent.index_number]
                            break
                    # 因为通讯距离问题，会有的无人机无法统计到所有的攻击无人机
                    elif attack_number < 4:
                        one_agent_boids_policy = self.boids_policy
                        now_agent_pos = agent.state.p_pos
                        now_agent_vel = agent.state.p_vel
                        action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos,
                                                                                    need_dist,
                                                                                    now_agent_vel, all_vel,
                                                                                    time_step)
                        break
        if action is None:
            # 保底，采用boids
            one_agent_boids_policy = self.boids_policy
            now_agent_pos = agent.state.p_pos
            now_agent_vel = agent.state.p_vel
            action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos,
                                                                        need_dist,
                                                                        now_agent_vel, all_vel,
                                                                        time_step)

        return action
