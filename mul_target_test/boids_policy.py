# -*- coding: utf-8 -*-
# @Time : 2022/6/7 下午6:52
# @Author :  wangshulei
# @FileName: boids_policy.py
# @Software: PyCharm
import copy
import math

import numpy as np


class boids_policy:
    def __init__(self,
                 agent_number,
                 agent_com_size,
                 max_vel,
                 rule_1_distance=4.0,
                 rule_2_distance=4.0,
                 rule_3_distance=0.5):
        self.agent_number = agent_number
        self.max_vel = max_vel
        self.agent_com_size = agent_com_size
        self.rule_1_distance = rule_1_distance
        self.rule_2_distance = rule_2_distance
        self.rule_3_distance = rule_3_distance

    def one_agent_apply_boids_rules(self, now_agent_pos, all_pos, need_dist, now_agent_vel,
                                    all_vel, time_step, need_Hover=True):
        """
        单个智能体应用boids规则
        :param need_Hover: 直翼无人机不能够悬浮，这里需要对其设置盘旋的算法
        :param now_agent_vel: 当前智能体的速度
        :param now_agent_pos: 当前智能体的位置
        :param need_dist: 当前智能体与其他能通讯智能体的距离
        :param all_pos: 当前智能体与其他能通讯智能体的位置
        :param all_vel: 当前智能体与其他能通讯智能体的速度
        :param time_step: 当前运行步数
        :return:
        """
        # 该智能体观测范围内的距离
        all_agent_dists = [copy.deepcopy(need_dist)]
        rule1_acc = []
        now_agent_dist = all_agent_dists[0]  # 以及和他们的距离
        x_mean = now_agent_pos[0]
        y_mean = now_agent_pos[1]
        number_rule1_agent = 0
        for near_agent_index in range(len(now_agent_dist)):
            if now_agent_dist[near_agent_index] < self.rule_1_distance:
                x_mean += all_pos[near_agent_index][0]
                y_mean += all_pos[near_agent_index][1]
                number_rule1_agent += 1
        x_mean /= (number_rule1_agent + 1)
        y_mean /= (number_rule1_agent + 1)
        now_agent_acc_x = x_mean - now_agent_pos[0]
        now_agent_acc_y = y_mean - now_agent_pos[1]
        rule1_acc.append(0.5 * np.array([now_agent_acc_x, now_agent_acc_y]))

        rule2_acc = []
        now_agent_dist = all_agent_dists[0]
        x_vel_mean = now_agent_vel[0]
        y_vel_mean = now_agent_vel[1]
        number_rule2_agent = 0
        for near_agent_index in range(len(now_agent_dist)):
            if now_agent_dist[near_agent_index] < self.rule_2_distance:
                x_vel_mean += all_vel[near_agent_index][0]
                y_vel_mean += all_vel[near_agent_index][1]
                number_rule2_agent += 1
        x_vel_mean /= (number_rule2_agent + 1)
        y_vel_mean /= (number_rule2_agent + 1)
        now_agent_acc_x = x_vel_mean - now_agent_vel[0]
        now_agent_acc_y = y_vel_mean - now_agent_vel[1]
        rule2_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))

        rule3_acc = []
        now_agent_dists = all_agent_dists[0]
        x_dist_mean = 0
        y_dist_mean = 0
        number_rule3_agent = 0
        for near_agent_index in range(len(now_agent_dist)):
            if now_agent_dists[near_agent_index] < self.rule_3_distance:
                # ? index(当前agent) -- near_agent_index(小于舒适距离的agent编号)
                x_dist_mean += (now_agent_pos[0] - all_pos[near_agent_index][0])
                y_dist_mean += (now_agent_pos[1] - all_pos[near_agent_index][1])
                number_rule3_agent += 1
        x_dist_mean /= (number_rule3_agent + 1)
        y_dist_mean /= (number_rule3_agent + 1)
        if x_dist_mean == 0 and y_dist_mean == 0:
            now_agent_acc_x = 0
            now_agent_acc_y = 0
        else:
            now_agent_acc_x = 1. / x_dist_mean
            now_agent_acc_y = 1. / y_dist_mean
        rule3_acc.append(2 * np.array([now_agent_acc_x, now_agent_acc_y]))

        rule_acc = np.sum([rule1_acc, rule2_acc, rule3_acc], axis=0)
        rule_acc = self.apply_max_acc(rule_acc)

        action = []
        # 得到盘旋动作
        if need_Hover:
            points = []
            for i in range(125):
                ang = 2 * math.pi * i / 125
                points.append((math.cos(ang) * 2, math.sin(ang) * 2))
            hover_x = 0.5 * np.clip(points[time_step % 125][0] - now_agent_pos[0], -1.0, 1.0)
            hover_y = 0.5 * np.clip(points[time_step % 125][1] - now_agent_pos[1], -1.0, 1.0)
            Hover_action = np.array([0, hover_x,
                                     0, hover_y, 0])
        else:
            Hover_action = np.array([0, 0, 0, 0, 0])
        for rule_acc_one_agent in rule_acc:
            if time_step:
                action.append(np.array([0, 2 * rule_acc_one_agent[0], 0, 2 * rule_acc_one_agent[1], 0]) + Hover_action)
            else:
                action.append(np.array([0, 2 * rule_acc_one_agent[0], 0, 2 * rule_acc_one_agent[1], 0]))
        return action[0]

    def apply_boids_rules(self, all_pos, all_vel, time_step):
        # 得到每个智能体观测范围内的智能体个数
        all_obs_agent_index = []
        all_agent_dists = []
        for index, pos in enumerate(all_pos):
            # 求出所有距离
            dists = [np.sqrt(np.sum(np.square(pos - other_pos)))
                     for other_pos in all_pos]
            # 得到在观测距离内的智能体
            all_agent_dists.append(copy.deepcopy(dists))
            obs_agent_index = []
            for dis_index, dist in enumerate(dists):
                if dist < self.agent_com_size and dist != 0:
                    obs_agent_index.append(dis_index)
            all_obs_agent_index.append(copy.deepcopy(obs_agent_index))
        # //规则一：靠近
        # //靠近 周围可见区域 的 所有点的中心 即：提供一个 靠近中心 的 加速度
        # //要求：
        # //1.找出区域内的点
        # //2.求出中心
        # //3.设定加速大小方向,到中心的距离d越大，加速度a越大；方向为点到中心的方向
        rule1_acc = []
        for index in range(self.agent_number):
            near_agent_indexes = all_obs_agent_index[index]
            now_agent_dist = all_agent_dists[index]
            now_agent_pos = all_pos[index]
            x_mean = now_agent_pos[0]
            y_mean = now_agent_pos[1]
            number_rule1_agent = 0
            for near_agent_index in near_agent_indexes:
                if now_agent_dist[near_agent_index] < self.rule_1_distance and now_agent_dist[near_agent_index] != 0:
                    x_mean += all_pos[near_agent_index][0]
                    y_mean += all_pos[near_agent_index][1]
                    number_rule1_agent += 1
            x_mean /= (number_rule1_agent + 1)
            y_mean /= (number_rule1_agent + 1)
            now_agent_acc_x = x_mean - now_agent_pos[0]
            now_agent_acc_y = y_mean - now_agent_pos[1]
            rule1_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))
            # rule1_acc.append(np.array([0, 0]))
        # rule1_acc = self.apply_max_acc(rule1_acc)

        # //规则二：对齐方向
        # //调整 方向，使方向尽量靠近 区域内所有点的 平均速度方向,即：提供一个 旋转速度方向 的 加速度
        # //要求：
        # //1.找出区域内的点
        # //2.求出平均速度方向
        # //3.设定加速大小和方向
        # // 当前速度方向和平均速度方向的夹角alpha越大，加速度a越大；方向为垂直于当前速度朝着平均速度的方向
        rule2_acc = []
        for index in range(self.agent_number):
            near_agent_indexes = all_obs_agent_index[index]
            now_agent_dist = all_agent_dists[index]
            now_agent_val = all_vel[index]
            x_vel_mean = now_agent_val[0]
            y_vel_mean = now_agent_val[1]
            number_rule2_agent = 0
            for near_agent_index in near_agent_indexes:
                if now_agent_dist[near_agent_index] < self.rule_2_distance and now_agent_dist[near_agent_index] != 0:
                    x_vel_mean += all_vel[near_agent_index][0]
                    y_vel_mean += all_vel[near_agent_index][1]
                    number_rule2_agent += 1
            x_vel_mean /= (number_rule2_agent + 1)
            y_vel_mean /= (number_rule2_agent + 1)
            now_agent_acc_x = x_vel_mean - now_agent_val[0]
            now_agent_acc_y = y_vel_mean - now_agent_val[1]
            rule2_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))
            # rule2_acc.append(np.array([0, 0]))
        # rule2_acc = self.apply_max_acc(rule2_acc)

        # //规则三：避免碰撞
        # //避免和区域内的点的距离小于最小可靠近距离min dis
        # //即：当离区域内某点的距离小于某个值时，开始施加一个加速度
        # //要求：
        # //1.找出区域内的点
        # //2.求出当前点到其他点的距离
        # //3.如果距离小于某个值时，就产生一个加速
        # // 加速度大小随着距离的减小而增大；
        # // 方向为其他点到当前位置点的方向
        rule3_acc = []
        for index in range(self.agent_number):
            near_agent_indexes = all_obs_agent_index[index]
            now_agent_dists = all_agent_dists[index]
            x_dist_mean = 0
            y_dist_mean = 0
            number_rule3_agent = 0
            for near_agent_index in near_agent_indexes:
                if now_agent_dists[near_agent_index] < self.rule_3_distance and \
                        now_agent_dists[near_agent_index] != 0:
                    # ? index(当前agent) -- near_agent_index(小于舒适距离的agent编号)
                    x_dist_mean += (all_pos[index][0] - all_pos[near_agent_index][0])
                    y_dist_mean += (all_pos[index][1] - all_pos[near_agent_index][1])
                    number_rule3_agent += 1
            x_dist_mean /= (number_rule3_agent + 1)
            y_dist_mean /= (number_rule3_agent + 1)
            if x_dist_mean == 0 and y_dist_mean == 0:
                now_agent_acc_x = 0
                now_agent_acc_y = 0
            else:
                now_agent_acc_x = 1. / x_dist_mean
                now_agent_acc_y = 1. / y_dist_mean
            rule3_acc.append(np.array([now_agent_acc_x, now_agent_acc_y]))
            # rule3_acc.append(np.array([0, 0]))
        # rule3_acc = self.apply_max_acc(rule3_acc)
        if time_step:
            rule_acc = np.sum([rule1_acc, rule2_acc, rule3_acc], axis=0)
            rule_acc = self.apply_max_acc(rule_acc)
        else:
            rule_acc = np.sum([rule3_acc, rule3_acc, rule3_acc], axis=0)
            rule_acc = self.apply_max_acc(rule_acc)
        action = []
        for index, rule_acc_one_agent in enumerate(rule_acc):
            if time_step:
                if 0 < time_step < 100:
                    action.append(np.array([0, rule_acc_one_agent[0], 0, rule_acc_one_agent[1] + 0.08, 0]))
                if 100 <= time_step < 200:
                    action.append(np.array([0, rule_acc_one_agent[0] + 0.08, 0, rule_acc_one_agent[1], 0]))
                if 200 <= time_step < 300:
                    action.append(np.array([0, rule_acc_one_agent[0], 0, rule_acc_one_agent[1] - 0.08, 0]))
                if 300 <= time_step < 400:
                    action.append(np.array([0, rule_acc_one_agent[0] - 0.08, 0, rule_acc_one_agent[1], 0]))
                if 400 <= time_step <= 500:
                    action.append(np.array([0, rule_acc_one_agent[0], 0, rule_acc_one_agent[1] + 0.08, 0]))
            else:
                action.append(np.array([0, rule_acc_one_agent[0], 0, rule_acc_one_agent[1], 0]))
        return action

    def apply_max_acc(self, rule_all_agent_acc):
        all_acc = [np.sqrt(np.sum(np.square(agent_acc))) for agent_acc in rule_all_agent_acc]
        max_acc = max(all_acc)
        new_acc = []
        for rule_agent_acc in rule_all_agent_acc:
            if max_acc == 0:
                new_acc.append(rule_agent_acc)
            else:
                ratio = 1 / max_acc
                new_acc.append(ratio * rule_agent_acc * 0.333333)
        return new_acc
