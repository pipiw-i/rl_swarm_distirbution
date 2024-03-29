# -*- coding: utf-8 -*-
# @Time : 2022/6/7 下午13:18
# @Author :  wangshulei
# @FileName: algo_mul_target_test.py
# @Software: PyCharm
"""
老版本的策略执行文件
"""
import copy

import numpy as np
from tqdm import tqdm

from mul_target_test.RL_policy import RL_policy
from mul_target_test.boids_policy import boids_policy
from mul_target_test.logs import Logs
from mul_target_test.mpe_mul_target_env import mpe_env

SEED = 65535
ACTION_SPAN = 0.5
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
                 boids_rule_3_distance=2.5,
                 grouping_ratio=0.5
                 ):
        self.agent_number = agent_number
        self.landmark_number = landmark_number
        self.load_file = load_file
        self.n_test_times = n_test_times
        self.grouping_ratio = grouping_ratio
        self.boids_policy = boids_policy(self.agent_number, agent_com_size=4, max_vel=1,
                                         rule_1_distance=boids_rule_1_distance,
                                         rule_2_distance=boids_rule_2_distance,
                                         rule_3_distance=boids_rule_3_distance
                                         )
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
        agent, attack_number, agent.com_agent_index, all_pos, \
            all_relative_position, all_attack_agent_position, need_dist, all_vel, obs_landmarks, \
            obs_landmarks_feature = obs
        # 这里首先对检测到的目标进行识别，如果不属于自己管辖的范围，则删除该目标
        # 这里进行目标识别,这里简单分成两组，以中心为分界线，在中心右侧的为一组，在中心左侧的为一组
        # 分别对目标的相同分类方式产生关联
        new_targets = []  # 最终要执行的目标
        new_targets_feature = []
        for obs_landmark_index, obs_landmark in enumerate(obs_landmarks):
            obs_feature = obs_landmarks_feature[obs_landmark_index]
            # 首先判断agent_index的类别
            if agent_index >= int(self.grouping_ratio * self.agent_number):
                if obs_feature >= int(self.grouping_ratio * self.landmark_number):
                    new_targets.append(obs_landmark)
                    new_targets_feature.append(obs_feature)
            else:
                if obs_feature < int(self.grouping_ratio * self.landmark_number):
                    new_targets.append(obs_landmark)
                    new_targets_feature.append(obs_feature)
        if agent.is_destroyed:
            target = agent.attack_goal_pos
            now_pos = agent.state.p_pos
            act = 0.5 * (target - now_pos)
            action = np.array([0, act[0], 0, act[1], 0])
        elif len(new_targets) == 0:
            one_agent_boids_policy = self.boids_policy
            now_agent_pos = agent.state.p_pos
            now_agent_vel = agent.state.p_vel
            action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos, need_dist,
                                                                        now_agent_vel, all_vel, time_step)
        else:
            if attack_number == 0:
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
                    action_boids = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos,
                                                                                      need_dist,
                                                                                      now_agent_vel, all_vel,
                                                                                      time_step)
                    action = [one_agent_rl_policy.get_rl_action(new_obs) + 2 * action_boids,
                              new_targets_feature[obs_index],
                              agent.index_number]
                    break
            else:
                for obs_index, obs_landmark in enumerate(new_targets):
                    if attack_number >= 4:
                        # 当执行攻击的无人机大于4时
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


def policy_run(env, number_agent, landmark_number, load_file, need_render, grouping_ratio):
    agent_policy = AgentPolicy(agent_number=number_agent,
                               landmark_number=landmark_number,
                               load_file=load_file,
                               grouping_ratio=grouping_ratio).init_policy()
    obs_n = env.mpe_env.reset()
    landmark_attack_agent_index_dic_list = []
    landmark_been_attacked_list = []
    reassign_goals = False
    gaols_finish = False
    agent_finish = False
    for t in tqdm(range(1000)):
        # 决策攻击的无人机
        if need_render:
            env.mpe_env.render()
        action_n = []
        for agent_index in range(number_agent):
            action_n.append(copy.deepcopy(agent_policy[agent_index](obs=obs_n[agent_index],
                                                                    time_step=t,
                                                                    agent_index=agent_index)))
        # 统计部分，根据action统计每个目标所攻击的无人机编号
        rl_agent_number = 0
        landmark_attack_agent_index_dic = {}
        real_action = []
        for action in action_n:
            # 说明该部分是由rl产生的
            if len(action) == 3:
                rl_agent_number += 1
                # 统计所有攻击决策的无人机
                rl_act, obs_landmarks_feature, agent_index_number = action
                landmark_pos = obs_landmarks_feature  # 使用元组，才可以使用字典
                if rl_act[0] > 0.5:
                    if landmark_attack_agent_index_dic.get(landmark_pos, 0) == 0:
                        landmark_attack_agent_index_dic[landmark_pos] = [agent_index_number]
                    else:
                        landmark_attack_agent_index_dic[landmark_pos].append(agent_index_number)
                real_action.append(copy.deepcopy(rl_act))
            else:
                real_action.append(copy.deepcopy(action))
        # print(f"landmark_attack_agent_index_dic is {landmark_attack_agent_index_dic}")
        new_obs_n = env.mpe_env.step([real_action, landmark_attack_agent_index_dic, reassign_goals,
                                      landmark_attack_agent_index_dic_list, grouping_ratio])

        attack_landmarks = list(landmark_attack_agent_index_dic.keys())
        for attack_landmark in attack_landmarks:
            attack_this_landmark_agent_index = landmark_attack_agent_index_dic.get(attack_landmark)
            attack_this_landmark_agent_number = len(attack_this_landmark_agent_index)

            if 1 <= attack_this_landmark_agent_number <= 2:
                if attack_landmark in landmark_been_attacked_list:
                    # 重分配出现的问题
                    if reassign_goals:
                        # 记录重分配的次数，如果次数过多，那么就适当放宽条件
                        for ind, landmark_item in enumerate(landmark_been_attacked_list):
                            if attack_landmark == landmark_item:
                                if 0 < len(attack_this_landmark_agent_index) <= 1:
                                    # 限制为1-2
                                    for a_i in attack_this_landmark_agent_index:
                                        # 防止重复
                                        if a_i in landmark_attack_agent_index_dic_list[ind][attack_landmark]:
                                            continue
                                        else:
                                            landmark_attack_agent_index_dic_list[ind][attack_landmark].append(a_i)
                                else:
                                    continue
                    continue
                else:
                    landmark_been_attacked_list.append(attack_landmark)
                    landmark_attack_agent_index_dic_list.append({attack_landmark: attack_this_landmark_agent_index})
        obs_n = new_obs_n
        if len(landmark_attack_agent_index_dic_list) == number_gaols:
            gaols_finish = True
        destroyed_number_count = 0
        for a, _, _, _, _, _, _, _, _, _ in new_obs_n:
            if a.is_destroyed:
                destroyed_number_count += 1
        if destroyed_number_count == number_agent:
            agent_finish = True
        if agent_finish:
            pass
        elif gaols_finish and not agent_finish:
            # print("现在进行重分配")
            reassign_goals = True

    return landmark_attack_agent_index_dic_list


def run_mpe(load_file, run_file='simple_search_team', number_agent=8,
            number_landmark=1, need_render=False, grouping_ratio=0.5):
    env = mpe_env(run_file, seed=SEED, number_agent=number_agent,
                  number_landmark=number_landmark, grouping_ratio=grouping_ratio)
    return env, number_agent, load_file, need_render


if __name__ == '__main__':
    group_ratio = 0.5  # 分成两组的比例
    distributions_logger = Logs(logger_name='distributions_logs', log_file_name='distributions_logs_6_6_17')
    result_logger = Logs(logger_name='result_logs', log_file_name='result_logs_6_6_17')
    for i in range(0, 4):
        all_distributions = {}
        number_UAVs = 8 + 8 * i
        number_gaols = 4 + 4 * i
        all_number_agent_attack_mean = []
        r_env, r_number_agent, r_load_file, r_need_render = run_mpe(
            load_file='../distribution_policy/run_distribution_e3_syn_para_double_random_6_6_17',
            run_file='simple_mul_target.py', number_agent=number_UAVs,
            number_landmark=number_gaols,
            need_render=True,
            grouping_ratio=group_ratio)
        print(f"环境初始化成功,number_UAVs:{number_UAVs},number_gaols:{number_gaols}")
        for j in range(2):
            distributions = policy_run(r_env, r_number_agent, number_gaols, r_load_file, r_need_render, group_ratio)
            all_goal_distributions = {}
            for distribution in distributions:
                for key_index, key_item in enumerate(list(distribution.keys())):
                    if all_goal_distributions.get(key_item, 0) == 0:
                        all_goal_distributions[key_item] = copy.deepcopy(distribution.get(key_item))
                    else:
                        for agent_index in distribution.get(key_item):
                            if agent_index in all_goal_distributions[key_item]:
                                continue
                            else:
                                all_goal_distributions[key_item].append(agent_index)
            # 统计这个轮次的平均值以及每个回合的分布情况
            # print(distributions)
            str_logs = f"当前轮数{j}_无人机数目{number_UAVs}_目标数目{number_gaols}\n"
            number_agent_attack_mean = 0
            final_list = list(all_goal_distributions.keys())
            final_list.sort()
            for key_index, key in enumerate(final_list):
                str_logs += str(key) + ':' + str(all_goal_distributions.get(key)) + '\n'
                if group_ratio * number_gaols - 1 == key:
                    str_logs += "-----------------------------------------------" + '\n'
                number_agent_attack_mean += len(all_goal_distributions.get(key))
            # 得到一轮测试每个目标的平均攻击数目
            number_agent_attack_mean /= number_gaols
            all_number_agent_attack_mean.append(number_agent_attack_mean)
            distributions_logger.add_logs(str_logs)
        str_logs = f"result_{number_UAVs}_{number_gaols}_50_times\n"
        mean_50 = 0
        for times, one_number_agent_attack_mean in enumerate(all_number_agent_attack_mean):
            str_logs += f"当前轮数{times},平均攻击次数{one_number_agent_attack_mean}\n"
            mean_50 += one_number_agent_attack_mean
        mean_50 /= 50
        str_logs += f"在无人机数目{number_UAVs}目标数目{number_gaols}下，平均攻击次数为{mean_50}\n"
        result_logger.add_logs(str_logs)
    print("程序运行完毕！")
