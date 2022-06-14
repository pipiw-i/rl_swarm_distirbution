# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午12:43
# @Author :  wangshulei
# @FileName: simple_distribution.py
# @Software: PyCharm
"""
使用训练好的无人机策略，使用任意数目的无人机进行目标分配上进行测试
"""
import copy

import numpy as np

from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario

cam_range = 4  # 视角范围


class Scenario(BaseScenario):
    def __init__(self,
                 num_agents=32,  # 160 rgb
                 num_landmarks=1,  # < 100 rgb
                 agent_size=0.05,
                 search_size=1,
                 com_size=4):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.agent_size = agent_size
        self.search_size = search_size
        self.com_size = com_size
        # 每个机器人的路径轨迹
        self.other_agent_pos = []
        # 历史中曾经探测到的目标位置
        self.landmark_get = []
        self.attack_number = 0
        self.move = False  # 移动执行攻击的无人机

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
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        self.other_agent_pos = []
        self.move = False
        self.attack_number = 0
        self.landmark_get.clear()  # 存储的历史位置清零
        world.assign_agent_colors()
        world.assign_landmark_colors()
        # set random initial states  随机初始状态
        for agent_index, agent in enumerate(world.agents):
            # 防止初始的随机地点会重叠到一起
            agent.history_landmark_position.clear()
            agent.benchmark_position.clear()
            agent.com_agent_index.clear()
            agent.attack = False
            agent.one_attack_number = 0
            agent.attack_rate = 0
            if len(self.other_agent_pos) == 0:
                agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                         for pos in self.other_agent_pos]
                while min(dists) < 0.1 * agent.search_size or max(dists) > agent.com_size:
                    agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                             for pos in self.other_agent_pos]
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.index_number = agent_index
            self.other_agent_pos.append(copy.deepcopy(agent.state.p_pos))
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.9 * np.random.uniform(-cam_range, +cam_range, world.dim_p)
            # landmark.state.p_pos = np.array([0.01, 0.01])
            landmark.state.p_vel = np.array([0, 0])

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
                        if (ag_benchmark_pos - obs_landmark < 0.005).all():
                            if_exist = True
                    if not if_exist:
                        for ag_benchmark_position in ag.benchmark_position:
                            obs_landmarks.append(copy.deepcopy(ag_benchmark_position))
            # 可以联系上的智能体,这里是能够直接联系的智能体，不考虑间接联系
            elif int(ag.name[-1]) in agent.com_agent_index:
                for ag_benchmark_pos in ag.benchmark_position:
                    if_exist = False
                    for obs_landmark in obs_landmarks:
                        if (ag_benchmark_pos - obs_landmark < 0.005).all():
                            if_exist = True
                    if not if_exist:
                        for ag_benchmark_position in ag.benchmark_position:
                            obs_landmarks.append(copy.deepcopy(ag_benchmark_position))
        if obs_landmarks:  # obs_landmarks 所有智能体能够检测到的目标
            for obs_landmark in obs_landmarks:
                is_exist = False
                for landmark_get in self.landmark_get:
                    if (obs_landmark - landmark_get < 0.00005).all():
                        is_exist = True
                        continue
                if not is_exist:
                    self.landmark_get.append(copy.deepcopy(obs_landmark))
        if self.landmark_get:
            for a in world.agents:
                a.history_landmark_position = self.landmark_get
        #       print(f"a.history_landmark_position is {a.history_landmark_position}")
        # print(f"self.landmark_get is {self.landmark_get}")
        return self.landmark_get

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
        # 该函数用来 ① 更新障碍物周围无人机的个数 ② 更新目标的速度(如果可以移动的话) ③ 更新能够联系上的智能体个数
        #          ④ 更新当前所有的无人机能够探测的目标位置(仅自己的范围)
        for landmark in world.landmarks:
            ag_number = 0  # 该障碍物周围智能体的个数
            # 该障碍物与其他所有智能体的距离
            landmark_dists = [np.sqrt(np.sum(np.square(landmark.state.p_pos - a.state.p_pos)))
                              for a in world.agents]
            for landmark_dist in landmark_dists:
                #  如果该障碍物的距离与智能体之间的距离小于探测距离
                if landmark_dist < 0.5 * self.search_size:
                    ag_number += 1
            landmark.landmark_around_agent_number = ag_number
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.array([0.05, 0])
        # 更新目前能够联系上的智能体编号以及能够检测到的目标
        for agent in world.agents:
            agent.com_agent_index.clear()
            agent.benchmark_position.clear()
            dists = [[np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))), l.state.p_pos]
                     for l in world.landmarks]
            for dist, pos in dists:
                if dist < agent.search_size:
                    agent.benchmark_position.append(pos)
            # 得到与其他智能体的距离，看是否能够通讯
            com_dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                         for a in world.agents]
            for index, com_dist in enumerate(com_dists):
                if com_dist != 0 and com_dist < agent.com_size:
                    agent.com_agent_index.append(index)
        attack_number = 0
        for other in world.agents:
            # 只在交流范围内的无人机有关，包括自己！
            # if np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos))) < agent.com_size:
            #     if other.attack:
            #         attack_number += 1
            #     other_pos.append(other.state.p_pos - agent.state.p_pos)
            # 取消观测限制，能够观测到区域内所有的智能体攻击情况，而不仅仅是观测范围内的，因为reward也是根据全部的无人机得到的。
            # 相当于通讯范围无限
            if other.attack:
                attack_number += 1
        if attack_number > 2:
            # attack_number = np.random.randint(3, 33, 1)[0]
            attack_number = attack_number
        self.attack_number = attack_number
        for other_a in world.agents:
            if other_a.attack:
                other_a.one_attack_number += 1

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
        # obs_landmarks = self.get_landmark(agent, world)
        obs_landmarks = world.landmarks
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
