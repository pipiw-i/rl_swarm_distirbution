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

cam_range = 2  # 视角范围


cam_range = 2  # 视角范围


class Scenario(BaseScenario):
    def __init__(self,
                 num_agents=2,  # 160 rgb
                 num_landmarks=1,  # < 100 rgb
                 agent_size=0.05,
                 search_size=0.5,
                 com_size=0.9):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.agent_size = agent_size
        self.search_size = search_size
        self.com_size = com_size
        # 每个机器人的路径轨迹
        self.other_agent_pos = []

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
        world.assign_agent_colors()
        world.assign_landmark_colors()
        # set random initial states  随机初始状态
        for agent_index, agent in enumerate(world.agents):
            # 防止初始的随机地点会重叠到一起
            if len(self.other_agent_pos) == 0:
                agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                         for pos in self.other_agent_pos]
                while min(dists) < agent.size:
                    agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                             for pos in self.other_agent_pos]
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            self.other_agent_pos.append(copy.deepcopy(agent.state.p_pos))
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = 0.8 * np.random.uniform(-cam_range, +cam_range, world.dim_p)
            landmark.state.p_pos = np.array([0, 0])
            landmark.state.p_vel = np.zeros(world.dim_p)

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

    def reward(self, agent, world, r=0):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # 不考虑相对于目标点的距离，目前进行探索策略的训练
        for ag in world.agents:
            # 如果出界，则奖励减小
            if ag.state.p_pos[0] > cam_range or ag.state.p_pos[0] < -cam_range:
                rew -= ag.com_size
            if ag.state.p_pos[1] > cam_range or ag.state.p_pos[1] < -cam_range:
                rew -= ag.com_size
            # 在区域内
            if -cam_range < ag.state.p_pos[0] < cam_range and -cam_range < ag.state.p_pos[1] < cam_range:
                for l in world.landmarks:
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                    # 如果搜索区出现了目标，则向目标区域靠近
                    if dist < agent.search_size:
                        rew -= dist  # 0.2
                    else:
                        rew -= agent.search_size  # 0.2
                dists = [np.sqrt(np.sum(np.square(ag.state.p_pos - a.state.p_pos)))
                         for a in world.agents]
                for dist in dists:
                    # 尽量不要重叠搜索区域
                    if ag.com_size > dist > 0.5 * ag.search_size:
                        rew += 1
        # 部分可观测的系统，智能体只能对观测范围内的目标做出反应
        if agent.collide:
            for a in world.agents:
                if a == agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos))) < agent.search_size:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
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
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + comm)
        return obs
