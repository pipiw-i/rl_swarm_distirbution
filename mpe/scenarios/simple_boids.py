# -*- coding: utf-8 -*-
# @Time : 2022/5/27 下午14:47
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


class Scenario(BaseScenario):
    def __init__(self,
                 num_agents=4,  # 160 rgb
                 num_boids_agent=50,
                 num_landmarks=2,  # < 100 rgb
                 agent_size=0.05,
                 search_size=1,
                 com_size=3):
        self.num_agents = num_agents
        self.num_boids_agent = num_boids_agent
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
        world.num_boids_agents = self.num_boids_agent
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
        # 加载 boids 智能体
        world.boids_agents = [Agent() for i in range(world.num_boids_agents)]
        for i, boids_agent in enumerate(world.boids_agents):
            boids_agent.name = 'boids_agent %d' % i
            boids_agent.collide = True
            boids_agent.silent = True
            boids_agent.size = self.agent_size
            boids_agent.search_size = self.search_size
            boids_agent.com_size = self.com_size
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
                while min(dists) < agent.search_size:
                    agent.state.p_pos = np.random.uniform(-cam_range, +cam_range, world.dim_p)
                    dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - pos)))
                             for pos in self.other_agent_pos]
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            self.other_agent_pos.append(copy.deepcopy(agent.state.p_pos))
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = 0.8 * np.random.uniform(-cam_range, +cam_range, world.dim_p)
            landmark.state.p_pos = np.array([0.01, 0.01])
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



    def get_landmark(self,agent,world):
        obs_landmarks = []
        for ag in world.agents:
            if ag == agent:
                if ag.benchmark_position:
                    if ag.benchmark_position in obs_landmarks:
                        continue
                    else:
                        obs_landmarks.append(ag.benchmark_position)
            # 可以联系上的智能体
            elif int(ag.name[-1]) in agent.com_agent_index:
                if ag.benchmark_position in obs_landmarks:
                    continue
                else:
                    if ag.benchmark_position:
                        obs_landmarks.append(ag.benchmark_position)
        return obs_landmarks

    def reward(self, agent, world, r=0):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.state.p_pos[0] > cam_range or agent.state.p_pos[0] < -cam_range:
            rew -= 2.5
        elif agent.state.p_pos[1] > cam_range or agent.state.p_pos[1] < -cam_range:
            rew -= 2.5
        # 在区域内
        elif -cam_range < agent.state.p_pos[0] < cam_range and -cam_range < agent.state.p_pos[1] < cam_range:
            # 先判断智能体之间的位置关系
            dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                        for a in world.agents]
            landmark = self.get_landmark(agent, world)  # 得到所有的能观测到的障碍物
            if landmark:
                for index, l in enumerate(landmark):
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l[0])))
                    # 如果搜索区出现了目标，则向目标区域靠近
                    rew -= 0.2 * dist
            else:
                rew -= 0.6
            for dist in dists:
                # 尽量不要重叠搜索区域
                if agent.com_size > dist or dist == 0:
                    rew += 0  # 在通讯范围内，则不变
                else:
                    # 若逃出通讯范围，则减0.2，迫使无人机停留在通讯范围内，但是此刻如果
                    # 观测到了障碍物
                    if_landmark = False
                    for l in landmark:
                        if l:
                            if_landmark = True
                    if not if_landmark:
                        rew -= 0.2
        # 部分可观测的系统，智能体只能对观测范围内的目标做出反应
        if agent.collide:
            for a in world.agents:
                if a == agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def get_obs_landmarks_pos(self, world):
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
        obs_landmarks = []
        for ag in world.agents:
            if ag == agent:
                if ag.benchmark_position:
                    if ag.benchmark_position in obs_landmarks:
                        continue
                    else:
                        obs_landmarks.append(ag.benchmark_position)
            # 可以联系上的智能体
            elif int(ag.name[-1]) in agent.com_agent_index:
                if ag.benchmark_position:
                    if ag.benchmark_position in obs_landmarks:
                        continue
                    else:
                        obs_landmarks.append(ag.benchmark_position)
        # 得到观测值
        for index, obs_landmark in enumerate(obs_landmarks):  # world.entities:
            entity_pos.append(obs_landmark[0] - agent.state.p_pos)
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
