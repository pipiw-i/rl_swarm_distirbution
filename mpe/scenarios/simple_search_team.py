# -*- coding: utf-8 -*-
# @Time : 2022/5/17 上午9:52
# @Author :  wangshulei
# @FileName: simple_search.py
# @Software: PyCharm
"""
无人机搜索的环境，在一个方形的环境中尽快的搜索环境，找到目标
"""
import copy

import numpy as np

from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario

cam_range = 4  # 视角范围


class Scenario(BaseScenario):
    def __init__(self,
                 num_agents=2,  # 160 rgb
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
        self.landmark_get.clear()  # 存储的历史位置清零
        world.assign_agent_colors()
        world.assign_landmark_colors()
        # set random initial states  随机初始状态
        for agent_index, agent in enumerate(world.agents):
            # 防止初始的随机地点会重叠到一起
            agent.history_landmark_position.clear()
            agent.benchmark_position.clear()
            agent.com_agent_index.clear()
            if len(self.other_agent_pos) == 0:
                agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                         for pos in self.other_agent_pos]
                while min(dists) < agent.search_size:
                    agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                             for pos in self.other_agent_pos]
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            self.other_agent_pos.append(copy.deepcopy(agent.state.p_pos))
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-cam_range, +cam_range, world.dim_p)
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
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.state.p_pos[0] > cam_range or agent.state.p_pos[0] < -cam_range:
            rew -= 2.5
        elif agent.state.p_pos[1] > cam_range or agent.state.p_pos[1] < -cam_range:
            rew -= 2.5
        # 在区域内
        elif -cam_range < agent.state.p_pos[0] < cam_range and -cam_range < agent.state.p_pos[1] < cam_range:
            # 先判断智能体之间的位置关系
            # dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
            #          for a in world.agents]
            # landmark = self.get_landmark(agent, world)  # 得到所有的能观测到的障碍物
            # for lm in world.landmarks:
            #     # 在能够检测到的障碍物里面
            #     for obs_lm in landmark:
            #         if (lm.state.p_pos - obs_lm < 0.005).all():
            #             if lm.landmark_around_agent_number == 2:
            #                 rew += 0.5
            # if landmark:
            #     for index, l in enumerate(landmark):
            #         # dist = np.mean([np.sqrt(np.sum(np.square(a.state.p_pos - l))) for a in world.agents], axis=0)
            #         dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l)))
            #         # 如果搜索区出现了目标，则向目标区域靠近
            #         rew -= 0.3 * dist
            # for dist in dists:
            #     if agent.com_size > dist or dist == 0:
            #         rew += 0  # 在通讯范围内，则不变
            #     else:
            #         # 若逃出通讯范围，则减0.8，迫使无人机停留在通讯范围内
            #         rew -= 0.8
            landmark = world.landmarks
            if landmark:
                for index, l in enumerate(landmark):
                    # dist = np.mean([np.sqrt(np.sum(np.square(a.state.p_pos - l))) for a in world.agents], axis=0)
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                    # 如果搜索区出现了目标，则向目标区域靠近
                    rew -= dist
        # 部分可观测的系统，智能体只能对观测范围内的目标做出反应
        if agent.collide:
            for a in world.agents:
                if a == agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 2
        return rew

    def get_obs_landmarks_pos(self, world):
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

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # obs_landmarks = self.get_landmark(agent, world)
        obs_landmarks = world.landmarks
        # 得到观测值
        for index, obs_landmark in enumerate(obs_landmarks):  # world.entities:
            entity_pos.append(obs_landmark.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            if np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos))) < agent.com_size:
                comm.append(other.state.c)
                # 与自身的相对位置以及相对速度
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel - agent.state.p_vel)
        # 这样obs就不定长了
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel + comm + entity_pos)
        return obs
