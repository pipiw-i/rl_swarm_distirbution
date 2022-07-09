# -*- coding: utf-8 -*-
# @Time : 2022/6/7 下午13:00
# @Author :  wangshulei
# @FileName: simple_mul_target.py
# @Software: PyCharm
"""
使用训练好的无人机策略，在多个目标上进行测试
"""
import copy

import numpy as np

from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
from mul_target_test.logs import Logs

cam_range = 4  # 视角范围


class Scenario(BaseScenario):
    def __init__(self,
                 need_log=False,
                 num_agents=32,  # 160 rgb
                 num_landmarks=16,  # < 100 rgb
                 agent_size=0.05,
                 search_size=1,
                 com_size=4):
        self.need_log = need_log
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.agent_size = agent_size
        self.search_size = search_size
        self.com_size = com_size
        self.logger = Logs('comm_logger', 'mul_target')
        # 其他位置信息
        self.other_agent_pos = []
        self.other_landmark_pos = []
        # 历史中曾经探测到的目标位置
        self.landmark_get = []
        self.attack_number = 0
        self.move = False  # 移动执行攻击的无人机
        self.grouping_ratio = 0

    def make_world(self):
        world = World()
        world.world_length = 200
        # set any world properties first
        world.dim_c = 2  # 二维
        world.num_agents = self.num_agents
        world.num_landmarks = self.num_landmarks  # 3
        world.collaborative = True  # 是否具有体积
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size
            agent.search_size = self.search_size
            agent.com_size = self.com_size
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = True
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        self.other_agent_pos = []
        self.other_landmark_pos = []
        self.move = False
        self.attack_number = 0
        self.landmark_get.clear()  # 存储的历史位置清零

        # set random initial states  随机初始状态
        for agent_index, agent in enumerate(world.agents):
            # 防止初始的随机地点会重叠到一起
            agent.history_landmark_position.clear()
            agent.benchmark_position.clear()
            agent.com_agent_index.clear()
            agent.attack = False
            agent.one_attack_number = 0
            agent.attack_rate = 0
            agent.is_destroyed = False
            agent.attack_goal = None
            agent.collide = True
            agent.is_decided = False
            agent.now_goal = None
            agent.now_goal_feature = None
            agent.landmark_feature.clear()
            agent.attack_goal_pos = None
            if len(self.other_agent_pos) == 0:
                agent.state.p_pos = np.random.uniform(- 4 * cam_range, - 2 * cam_range, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.random.uniform(- 4 * cam_range, - 2 * cam_range, world.dim_p)
                dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                         for pos in self.other_agent_pos]
                while min(dists) < 0.1 * agent.search_size or max(dists) > agent.com_size:
                    agent.state.p_pos = np.random.uniform(- 4 * cam_range, - 2 * cam_range, world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                             for pos in self.other_agent_pos]
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.index_number = agent_index
            self.other_agent_pos.append(copy.deepcopy(agent.state.p_pos))
        for i, landmark in enumerate(world.landmarks):
            landmark.been_attacked = False
            if len(self.other_landmark_pos) == 0:
                landmark.state.p_pos = np.random.uniform(- 0.5 * cam_range, + 0.5 * cam_range, world.dim_p)
            else:
                landmark.state.p_pos = np.random.uniform(- 0.5 * cam_range, + 0.5 * cam_range, world.dim_p)
                dists = [np.sqrt(np.sum(np.square(landmark.state.p_pos - pos)))
                         for pos in self.other_landmark_pos]
                while min(dists) < 0.5:
                    landmark.state.p_pos = np.random.uniform(- 0.5 * cam_range, + 0.5 * cam_range, world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(landmark.state.p_pos - pos)))
                             for pos in self.other_landmark_pos]
            # landmark.state.p_pos = np.array([0.01, 0.01])
            landmark.state.p_vel = np.array([0, 0])
            landmark.feature = i
            self.other_landmark_pos.append(copy.deepcopy(landmark.state.p_pos))
        world.assign_rl_agent_colors(self.grouping_ratio)
        world.assign_landmark_colors(self.grouping_ratio)

    def set_agent_landmark_numbers(self, agent_numbers, landmark_numbers, grouping_ratio):
        self.num_agents = agent_numbers
        self.num_landmarks = landmark_numbers
        self.grouping_ratio = grouping_ratio

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def get_landmark(self, agent, world):
        obs_landmarks = []
        for ag in world.agents:
            if ag == agent:
                for ag_benchmark_pos in ag.benchmark_position:
                    if_exist = False
                    for obs_landmark in obs_landmarks:
                        if (abs(ag_benchmark_pos - obs_landmark) < 0.005).all():
                            if_exist = True
                    if not if_exist:
                        for ag_benchmark_position in ag.benchmark_position:
                            obs_landmarks.append(copy.deepcopy(ag_benchmark_position))
            # 可以联系上的智能体,这里是能够直接联系的智能体，不考虑间接联系
            elif int(ag.name[-1]) in agent.com_agent_index:
                for ag_benchmark_pos in ag.benchmark_position:
                    if_exist = False
                    for obs_landmark in obs_landmarks:
                        if (abs(ag_benchmark_pos - obs_landmark) < 0.005).all():
                            if_exist = True
                    if not if_exist:
                        for ag_benchmark_position in ag.benchmark_position:
                            obs_landmarks.append(copy.deepcopy(ag_benchmark_position))
        if obs_landmarks:  # obs_landmarks 所有智能体能够检测到的目标
            for obs_landmark in obs_landmarks:
                is_exist = False
                for landmark_get in self.landmark_get:
                    if (abs(obs_landmark - landmark_get) < 0.00005).all():
                        is_exist = True
                        continue
                if not is_exist:
                    self.landmark_get.append(copy.deepcopy(obs_landmark))
        if self.landmark_get:
            for a in world.agents:
                a.history_landmark_position = self.landmark_get
        # print(f"self.landmark_get is {self.landmark_get}")
        r_obs_landmarks = copy.deepcopy(obs_landmarks)
        return r_obs_landmarks

    def get_landmark_from_feature(self, agent, world):
        obs_landmarks = []
        for ag in world.agents:
            if ag == agent:
                for ag_landmark_feature in ag.landmark_feature:
                    if_exist = False
                    for obs_landmark in obs_landmarks:
                        if ag_landmark_feature == obs_landmark:
                            if_exist = True
                    if not if_exist:
                        for new_ag_landmark_feature in ag.landmark_feature:
                            obs_landmarks.append(copy.deepcopy(new_ag_landmark_feature))
            # 可以联系上的智能体,这里是能够直接联系的智能体，不考虑间接联系
            elif int(ag.name[-1]) in agent.com_agent_index:
                for ag_landmark_feature in ag.landmark_feature:
                    if_exist = False
                    for obs_landmark in obs_landmarks:
                        if ag_landmark_feature == obs_landmark:
                            if_exist = True
                    if not if_exist:
                        for new_ag_landmark_feature in ag.landmark_feature:
                            obs_landmarks.append(copy.deepcopy(new_ag_landmark_feature))
        if obs_landmarks:  # obs_landmarks 所有智能体能够检测到的目标
            for obs_landmark in obs_landmarks:
                is_exist = False
                for landmark_get in self.landmark_get:
                    if obs_landmark == landmark_get:
                        is_exist = True
                        continue
                if not is_exist:
                    self.landmark_get.append(copy.deepcopy(obs_landmark))
        if self.landmark_get:
            for a in world.agents:
                a.history_landmark_position = self.landmark_get
        # print(f"self.landmark_get is {self.landmark_get}")
        obs_landmarks.sort()
        r_obs_landmarks = copy.deepcopy(obs_landmarks)
        r_obs_landmarks_pos = []
        for landmark in world.landmarks:
            if landmark.feature in r_obs_landmarks:
                r_obs_landmarks_pos.append(copy.deepcopy(landmark.state.p_pos))
        return r_obs_landmarks_pos, r_obs_landmarks

    def reward(self, agent, world):
        rew = 0
        attack_agent_number = 0
        for l in world.landmarks:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            rew -= dist
        if self.attack_number == 2:
            rew += 2
        # 对超过2个无人机的行为进行惩罚
        elif self.attack_number > 2:
            rew -= 0.4 * (self.attack_number - 2)
        # 对小于两个无人机的行为进行惩罚
        elif self.attack_number == 0:
            rew -= 1.6
        elif self.attack_number == 1:
            rew -= 0.4
        if agent.collide:
            for a in world.agents:
                if a == agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def get_obs_landmarks_pos(self, world):
        # 该函数用来 ③ 更新能够联系上的智能体个数
        #           ④ 更新当前所有的无人机能够探测的目标位置(仅自己的范围)
        # print("更新智能体状态，探测目标以及联系智能体编号")
        for landmark in world.landmarks:
            # landmark.state.p_vel = np.random.uniform(-0.1, +0.1, world.dim_p)
            landmark.state.p_vel = np.array([-0.01, -0.01])
        for agent in world.agents:
            agent.com_agent_index.clear()
            agent.benchmark_position.clear()
            agent.landmark_feature.clear()
            dists = [[np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))),
                      l.state.p_pos,
                      l.been_attacked,
                      l.feature]
                     for l in world.landmarks]
            for dist, pos, been_attacked, l_feature in dists:
                if dist < agent.search_size and not been_attacked:
                    agent.benchmark_position.append(pos)
                    agent.landmark_feature.append(l_feature)

            # 得到与其他智能体的距离，看是否能够通讯
            com_dists = [[np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos))), a.is_destroyed]
                         for a in world.agents]
            for index, com_dist_is_destroyed in enumerate(com_dists):
                com_dist = com_dist_is_destroyed[0]
                is_destroyed = com_dist_is_destroyed[-1]
                if com_dist != 0 and com_dist < agent.com_size and not is_destroyed:
                    agent.com_agent_index.append(index)

    def get_attack_number(self, world):
        attack_index = []
        for agent in world.agents:
            if agent.attack:
                attack_index.append(agent.index_number)
        return self.attack_number, attack_index

    def get_attack_agent_index(self, world, n_game):
        attack_index = []
        for agent in world.agents:
            agent.attack_rate = agent.one_attack_number / n_game
            if agent.attack_rate >= 0.8:
                attack_index.append(agent.index_number)
        return attack_index

    def set_move(self):
        self.move = True

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # 这里选取能够观测到的智能体,obs_landmarks 内就是当前智能体与他直接联系的智能体所能够观测到的障碍物
        obs_landmarks = self.get_landmark(agent, world)
        obs_landmarks.sort()  # 采用相同的算法，使得每个智能体观测到的多个目标能够顺序一致
        # obs_landmarks = world.landmarks
        # 得到观测值
        for index, obs_landmark in enumerate(obs_landmarks):  # world.entities:
            entity_pos.append(obs_landmark.state.p_pos - agent.state.p_pos)
        other_pos = []
        for other in world.agents:
            if not self.move:
                # 测试的时候，只与参与攻击的无人机进行交互，因为只有进行攻击的无人机才会移动
                if self.attack_number == 0:  # 说明全都没有攻击决策，那么就是重新决策
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                else:
                    # 如果攻击的大于四，那么要从所有能够攻击的无人机中，重新选择出更好的。
                    if self.attack_number >= 4:
                        if other.attack:
                            other_pos.append(other.state.p_pos - agent.state.p_pos)
                    else:
                        # 反正用不到了，这里采用其他所有的无人机位置
                        other_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
        mean_other_pos = np.mean(other_pos, axis=0)
        # 可能他的周围根本没有飞行器，那么就以他为中心
        obs = np.concatenate([np.array([self.attack_number])] + [agent.state.p_pos] + [mean_other_pos] + entity_pos)
        return obs

    def test_observation(self, agent, world, reassign_goals):
        # 这里选取能够观测到的智能体,obs_landmarks 内就是当前智能体与他直接联系的智能体所能够观测到的障碍物
        if agent.attack:
            agent.is_decided = True
        else:
            agent.is_decided = False
            agent.now_goal = None
        if not agent.is_decided:
            # obs_landmarks = self.get_landmark(agent, world)
            obs_landmarks, obs_landmarks_feature = self.get_landmark_from_feature(agent, world)
            if len(obs_landmarks) > 2:
                # np.sort(obs_landmarks)  # 采用相同的算法，使得每个智能体观测到的多个目标能够顺序一致
                agent.now_goal = [copy.deepcopy(obs_landmarks[0])]
                agent.now_goal_feature = [copy.deepcopy(obs_landmarks_feature[0])]
                obs_landmarks = copy.deepcopy(agent.now_goal)
                obs_landmarks_feature = copy.deepcopy(agent.now_goal_feature)
            else:
                agent.now_goal = obs_landmarks
                agent.now_goal_feature = obs_landmarks_feature
        else:
            obs_landmarks = agent.now_goal
            obs_landmarks_feature = agent.now_goal_feature
        # print(f"obs_landmarks is {obs_landmarks}")
        all_pos = []
        all_relative_position = []
        all_attack_agent_position = []
        all_vel = []
        need_dist = []
        attack_number = 0
        dists = [[np.sqrt(np.sum(np.square(agent.state.p_pos - w_agent.state.p_pos))), w_agent]
                 for w_agent in world.agents]
        for dist, w_agent in dists:
            if w_agent.is_destroyed:
                continue
            # dist == 0 只有其本身
            if w_agent.attack and dist == 0:
                attack_number += 1
                all_attack_agent_position.append(copy.deepcopy(w_agent.state.p_pos - agent.state.p_pos))
            elif not w_agent.attack and dist == 0:
                all_attack_agent_position.append(copy.deepcopy(w_agent.state.p_pos - agent.state.p_pos))
            # 判断其他的可联系智能体
            elif dist < agent.com_size and dist != 0:
                all_pos.append(copy.deepcopy(w_agent.state.p_pos))
                all_relative_position.append(copy.deepcopy(w_agent.state.p_pos - agent.state.p_pos))
                # 区分目标,统计相同目标的rl观测参数，包括攻击数目，攻击无人机的位置，以及自身的位置，目标位置(这里是全部的)
                if agent.index_number >= int(self.grouping_ratio * self.num_agents):
                    if w_agent.index_number >= self.num_agents // 2:
                        if w_agent.attack:
                            attack_number += 1
                            all_attack_agent_position.append(copy.deepcopy(w_agent.state.p_pos - agent.state.p_pos))
                else:
                    if w_agent.index_number < int(self.grouping_ratio * self.num_agents):
                        if w_agent.attack:
                            attack_number += 1
                            all_attack_agent_position.append(copy.deepcopy(w_agent.state.p_pos - agent.state.p_pos))
                all_vel.append(copy.deepcopy(w_agent.state.p_vel))
                need_dist.append(copy.deepcopy(dist))

        if len(obs_landmarks) == 0:
            # 不能直接检测到目标，不统计周围的攻击数目
            attack_number = 0
        # 在这里更改决策终止条件，当这个智能体本身的通讯发现其攻击个数在合适范围的时候，就会发出信号，停止下一次迭代更新
        if reassign_goals:
            offset = 1
        else:
            offset = 0
        if 1 - offset <= attack_number <= 2 - offset and agent.attack and not agent.is_destroyed:
            return_obs_landmarks_feature = []
            if agent.index_number < int(self.grouping_ratio * self.num_agents):
                for obs_landmarks_feature_i in obs_landmarks_feature:
                    if obs_landmarks_feature_i < int(self.grouping_ratio * self.num_landmarks):
                        return_obs_landmarks_feature.append(obs_landmarks_feature_i)

            else:
                for obs_landmarks_feature_i in obs_landmarks_feature:
                    if obs_landmarks_feature_i >= int(self.grouping_ratio * self.num_landmarks):
                        return_obs_landmarks_feature.append(obs_landmarks_feature_i)
            continue_decide_msg = [True, agent, return_obs_landmarks_feature]
        else:
            continue_decide_msg = [False]
        str_logs = f"当前智能体编号:{agent.index_number}\n " \
                   f"是否正在攻击:{agent.attack}\n " \
                   f"该无人机是否已经摧毁:{agent.is_destroyed}\n" \
                   f"该无人机摧毁的目标位置:{agent.attack_goal}\n" \
                   f"攻击数目:{attack_number}\n" \
                   f"当前智能体能够联系的智能体编号:{agent.com_agent_index}\n" \
                   f"当前智能体能够联系的智能体位置:{all_pos}\n" \
                   f"当前智能体能够联系的智能体与自己的相对位置:{all_relative_position}\n" \
                   f"当前智能体能够联系的攻击智能体位置:{all_attack_agent_position}\n" \
                   f"当前智能体能够联系的智能体的距离:{need_dist}\n" \
                   f"当前智能体能够联系的智能体速度:{all_vel}\n" \
                   f"当前无人机本身能够检测到的目标:{agent.benchmark_position}" \
                   f"当前智能体能够检测到的目标列表(位置):{obs_landmarks}\n"
        if self.need_log:
            self.logger.add_logs(str_logs)
        for landmark in world.landmarks:
            if landmark.feature == agent.attack_goal:
                agent.attack_goal_pos = landmark.state.p_pos
        return [agent, attack_number, agent.com_agent_index, all_pos,
                all_relative_position, all_attack_agent_position, need_dist, all_vel, obs_landmarks,
                obs_landmarks_feature, continue_decide_msg]
