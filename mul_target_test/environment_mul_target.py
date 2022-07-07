import copy

import gym
import numpy as np
from gym import spaces

from mpe.multi_discrete import MultiDiscrete

# update bounds to center around agent
cam_range = 4  # 视角范围


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None, land_callback=None,
                 get_attack_number_callback=None,
                 shared_viewer=True, discrete_action=True):

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback
        self.land_callback = land_callback
        self.get_attack_number_callback = get_attack_number_callback
        # 轨迹信息
        self.traj = None
        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        # self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)  # [-1,1]
            if agent.movable:
                total_action_space.append(u_action_space)

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(
                    world.dim_c,), dtype=np.float32)  # [0,1]

            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
            agent.action.c = np.zeros(self.world.dim_c)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n_landmark_attack_agent_index_dic):
        action_n, landmark_attack_agent_index_dic, reassign_goals, \
            landmark_attack_agent_index_dic_list,grouping_ratio = action_n_landmark_attack_agent_index_dic
        attack_landmarks = list(landmark_attack_agent_index_dic.keys())
        all_landmarks = self.world.landmarks
        self.agents = self.world.policy_agents
        # 根据对目标物的攻击，更改目标以及无人机的状态
        for world_landmark in all_landmarks:
            attack_agent_number = 0
            for landmark_agent in landmark_attack_agent_index_dic_list:
                attack_landmark = list(landmark_agent.keys())[0]
                attack_this_landmark_agent_index = []
                attack_landmark = np.array(attack_landmark)
                if world_landmark.feature == attack_landmark:
                    attack_agent_number += len(list(landmark_agent.values())[0])
                    for agent_index in list(landmark_agent.values())[0]:
                        attack_this_landmark_agent_index.append(agent_index)
                    if 0 < attack_agent_number <= 3:
                        world_landmark.been_attacked = True
                        if attack_agent_number == 1:
                            if world_landmark.feature < int(grouping_ratio * len(all_landmarks)):
                                world_landmark.color_list = [np.array([1, 1, 1]),
                                                             np.array([0.75, 0.25, 0.25]),
                                                             np.array([1, 1, 1]),
                                                             np.array([1, 1, 1])]
                            else:
                                world_landmark.color_list = [np.array([1, 1, 1]),
                                                             np.array([0.75, 0., 0.75]),
                                                             np.array([1, 1, 1]),
                                                             np.array([1, 1, 1])]

                        if attack_agent_number == 2:
                            if world_landmark.feature < int(grouping_ratio * len(all_landmarks)):
                                world_landmark.color_list = [np.array([1, 1, 1]),
                                                             np.array([0.75, 0.25, 0.25]),
                                                             np.array([1, 1, 1]),
                                                             np.array([0.75, 0.25, 0.25])]
                            else:
                                world_landmark.color_list = [np.array([1, 1, 1]),
                                                             np.array([0.75, 0., 0.75]),
                                                             np.array([1, 1, 1]),
                                                             np.array([0.75, 0., 0.75])]

                        if attack_agent_number == 3:
                            if world_landmark.feature < int(grouping_ratio * len(all_landmarks)):
                                world_landmark.color_list = [np.array([1, 1, 1]),
                                                             np.array([0.75, 0.25, 0.25]),
                                                             np.array([0.75, 0.25, 0.25]),
                                                             np.array([0.75, 0.25, 0.25])]
                            else:
                                world_landmark.color_list = [np.array([1, 1, 1]),
                                                             np.array([0.75, 0., 0.75]),
                                                             np.array([0.75, 0., 0.75]),
                                                             np.array([0.75, 0., 0.75])]

                for world_agent in self.agents:
                    if world_agent.index_number in attack_this_landmark_agent_index:
                        world_agent.is_destroyed = True
                        # 更新该无人机击毁的目标位置
                        for landmark in all_landmarks:
                            if landmark.feature == attack_landmark:
                                world_agent.attack_goal = landmark.feature
                        world_agent.collide = False
                        if world_agent.index_number < int(grouping_ratio * len(self.agents)):
                            world_agent.color = np.array([0.99, 0.25, 0.25])
                        else:
                            world_agent.color = np.array([0.99, 0.99, 0.25])
        # 根据目标分配的情况，决定是否重分配
        if reassign_goals:
            # 对于已分配的所有目标来说
            for landmark_agent in landmark_attack_agent_index_dic_list:
                landmark_d = list(landmark_agent.keys())[0]
                agent_d = list(landmark_agent.values())[0]
                if len(agent_d) == 1:
                    for landmark in all_landmarks:
                        if landmark.feature == np.array(landmark_d):
                            landmark.been_attacked = False

        obs_n = []
        # 根据第一个动作值，来决定后续的动作
        new_action_n = []
        for action_index, action in enumerate(action_n):
            if action[0] < 0.5:
                action = 0.8 * action
                self.agents[action_index].attack = False
            else:
                action = 0.8 * action
                self.agents[action_index].attack = True
            new_action_n.append(copy.deepcopy(action))
        self._get_landmark_pos()
        for i, agent in enumerate(self.agents):
            self._set_action(new_action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()  # core.step()  更新执行动作后的坐标
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
        return obs_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.traj = self.reset_callback(self.world)  # 获得初始坐标
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        self._get_landmark_pos()
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # 获取攻击个数
    def _get_attack_number(self):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.get_attack_number_callback()

    def _get_landmark_pos(self):
        if self.land_callback is None:
            return np.zeros(0)
        return self.land_callback(self.world)

    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]
        if agent.movable:
            # 交流信息
            if action[0][0] > 0:
                agent.action.c = 1.
            else:
                agent.action.c = 0.
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
                d = self.world.dim_p
            else:
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                    d = 5
                else:
                    if self.force_discrete_action:
                        p = np.argmax(action[0][0:self.world.dim_p])
                        action[0][:] = 0.0
                        action[0][p] = 1.0
                    agent.action.u = action[0][0:self.world.dim_p]
                    d = self.world.dim_p

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            if (not agent.silent) and (not isinstance(action_space, MultiDiscrete)):
                action[0] = action[0][d:]
            else:
                action = action[1:]

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]

            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
        self.passing_area = None

    def render(self, mode='human', close=False):
        if close:
            # close any existic renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''  # 输出交流信息
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' +
                                agent.name + ': ' + word + '   ')
            # print(message)
        # 因为是采用了shared_viewer，所以这里只有一个，如果不是，则有很多个
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from mpe import rendering  # 加载rendering文件
                self.viewers[i] = rendering.Viewer(700, 700)  # 建立一个window窗口，作为显示界面(白色画布)
        # create rendering geometry  创建渲染几何体
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering  仅当我们需要时才从gym导入渲染（对于headless machines不导入）
            from mpe import rendering  # 加载rendering文件
            self.render_geoms = []
            self.render_geoms_xform = []

            self.comm_geoms = []

            # geom = rendering.Line(start=(0.0, 0.0), end=(1.0, 0.0))
            # self.render_geoms.append(geom)

            for entity in self.world.entities:  # 对于所有的物体，包括智能体以及障碍物
                geom = None
                xform = None
                entity_comm_geoms = []
                if 'agent' in entity.name:
                    geom = rendering.make_circle(1.5 * entity.size)  # 画一个size大小的圆
                    xform = rendering.Transform()  # 建立一个位置的映射，使画在中心的圆能够映射到
                    # 探索区域
                    geom_search = rendering.make_circle(entity.search_size, filled=False)
                    geom_search.add_attr(xform)
                    # geom_search.set_color(0.75, 0.25, 0.25, alpha=0.5)
                    geom_search.set_color(0.75, 0.25, 0.25, alpha=0)
                    # 通信区域
                    geom_com = rendering.make_circle(entity.com_size, filled=False)
                    geom_com.add_attr(xform)
                    # geom_com.set_color(0.25, 0.25, 0.75, alpha=0.5)
                    geom_com.set_color(0.25, 0.25, 0.75, alpha=0)
                    geom.set_color(*entity.color, alpha=0.5)  # 对前面建立的圆加上color
                    self.render_geoms.append(geom_search)
                    self.render_geoms.append(geom_com)
                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)

                elif 'landmark' in entity.name:
                    geom = rendering.make_capsule(4 * entity.size, 4 * entity.size)  # 画一个size大小的圆
                    xform = rendering.Transform()  # 建立一个位置的映射，使画在中心的圆能够映射到
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            for wall in self.world.walls:
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                geom = rendering.make_polygon(corners)
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                self.render_geoms.append(geom)

            # add geoms to viewer
            # for viewer in self.viewers:
            #     viewer.geoms = []
            #     for geom in self.render_geoms:
            #         viewer.add_geom(geom)
            # 将要显示的东西加入到窗口中
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            if self.shared_viewer:
                all_agent_pos = np.zeros(self.world.dim_p)
                all_landmark_pos = np.zeros(self.world.dim_p)
                landmark_number = 0
                agent_number = 0
                for viewer_landmark in self.world.landmarks:
                    all_landmark_pos += viewer_landmark.state.p_pos
                    landmark_number += 1
                all_landmark_pos /= landmark_number
                for viewer_agent in self.agents:
                    all_agent_pos += viewer_agent.state.p_pos
                    agent_number += 1
                all_agent_pos /= agent_number
            else:
                all_agent_pos = np.zeros(self.world.dim_p)
                all_landmark_pos = np.zeros(self.world.dim_p)
            mid_pos = (all_agent_pos + all_landmark_pos)/2
            mid_pos_range = np.sqrt(np.sum(np.square(all_agent_pos - all_landmark_pos)))
            if i == 0:
                self.viewers[i].set_bounds(mid_pos[0] - max(0.5 * mid_pos_range, 1.5 * cam_range),
                                           mid_pos[0] + max(0.5 * mid_pos_range, 1.5 * cam_range),
                                           mid_pos[1] - max(0.5 * mid_pos_range, 1.5 * cam_range),
                                           mid_pos[1] + max(0.5 * mid_pos_range, 1.5 * cam_range))
            # update geometry positions
            agent_number = 0
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)  # 设置转换关系，将智能体的圆圈放到他本身的位置上
                if 'agent' in entity.name:
                    self.render_geoms[3 * e + 2].set_color(*entity.color, alpha=1.0)  # 设置颜色
                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
                    agent_number = e
                elif 'landmark' in entity.name:  # landmark
                    # self.render_geoms[2 * agent_number + e + 2].set_color(*entity.color, alpha=1.0)
                    for index, gs in enumerate(self.render_geoms[2 * agent_number + e + 2].gs):
                        gs.set_color(*entity.color_list[index], alpha=1.0)  # 对前面建立的圆加上color
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))
        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx
