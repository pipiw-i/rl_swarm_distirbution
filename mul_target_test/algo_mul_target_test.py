# -*- coding: utf-8 -*-
# @Time : 2022/6/7 下午13:18
# @Author :  wangshulei
# @FileName: algo_mul_target_test.py
# @Software: PyCharm
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
                 load_file='../distribution_policy/run_distribution_e3_syn_para_double_random_6_6_10',
                 n_test_times=6000,
                 boids_rule_1_distance=4.0,
                 boids_rule_2_distance=4.0,
                 boids_rule_3_distance=1.5
                 ):
        self.agent_number = agent_number
        self.load_file = load_file
        self.n_test_times = n_test_times
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
            all_relative_position, all_attack_agent_position, need_dist, all_vel, obs_landmarks = obs
        if agent.is_destroyed:
            target = agent.attack_goal
            now_pos = agent.state.p_pos
            act = 0.1 * (target - now_pos)
            action = np.array([0, act[0], 0, act[1], 0])
        elif len(obs_landmarks) == 0:
            one_agent_boids_policy = self.boids_policy
            now_agent_pos = agent.state.p_pos
            now_agent_vel = agent.state.p_vel
            action = one_agent_boids_policy.one_agent_apply_boids_rules(now_agent_pos, all_pos, need_dist,
                                                                        now_agent_vel, all_vel, time_step)
        else:
            if attack_number == 0:
                for obs_landmark in obs_landmarks:
                    one_agent_rl_policy = self.rl_policy
                    entity_pos = []
                    if len(all_relative_position) == 0:
                        mean_other_pos = np.mean([agent.state.p_pos], axis=0)
                    else:
                        mean_other_pos = np.mean(all_relative_position, axis=0)
                    entity_pos.append(obs_landmark - agent.state.p_pos)
                    new_obs = np.concatenate(
                        [np.array([attack_number])] + [agent.state.p_pos] + [mean_other_pos] + entity_pos)
                    action = [one_agent_rl_policy.get_rl_action(new_obs), obs_landmark, agent.index_number]
                    break
            else:
                for obs_landmark in obs_landmarks:
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
                            # 只对攻击的无人机继续采用rl策略
                            one_agent_rl_policy = self.rl_policy
                            entity_pos = []
                            mean_other_pos = np.mean(all_attack_agent_position, axis=0)
                            entity_pos.append(obs_landmark - agent.state.p_pos)
                            new_obs = np.concatenate(
                                [np.array([attack_number])] + [agent.state.p_pos] + [mean_other_pos] + entity_pos)
                            action = [one_agent_rl_policy.get_rl_action(new_obs), obs_landmark, agent.index_number]
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
        return action


def policy_run(env, number_agent, load_file, need_render):
    agent_policy = AgentPolicy(agent_number=number_agent, load_file=load_file).init_policy()
    obs_n = env.mpe_env.reset()
    landmark_attack_agent_index_dic_list = []
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
                rl_act, landmark, agent_index_number = action
                landmark_pos = (landmark[0], landmark[1])  # 使用元组，才可以使用字典
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
                                      landmark_attack_agent_index_dic_list])

        attack_landmarks = list(landmark_attack_agent_index_dic.keys())
        for attack_landmark in attack_landmarks:
            attack_this_landmark_agent_index = landmark_attack_agent_index_dic.get(attack_landmark)
            attack_this_landmark_agent_number = len(attack_this_landmark_agent_index)
            if 0 < attack_this_landmark_agent_number <= 3:
                landmark_attack_agent_index_dic_list.append(landmark_attack_agent_index_dic)
        obs_n = new_obs_n
        if len(landmark_attack_agent_index_dic_list) == number_gaols:
            gaols_finish = True
        destroyed_number_count = 0
        for a, _, _, _, _, _, _, _, _ in new_obs_n:
            if a.is_destroyed:
                destroyed_number_count += 1
        if destroyed_number_count == number_agent:
            agent_finish = True
        if agent_finish:
            pass
        elif gaols_finish and not agent_finish:
            reassign_goals = True

    return landmark_attack_agent_index_dic_list


def run_mpe(load_file, run_file='simple_search_team', number_agent=8, number_landmark=1, need_render=False):
    env = mpe_env(run_file, seed=SEED, number_agent=number_agent, number_landmark=number_landmark)
    return env, number_agent, load_file, need_render


if __name__ == '__main__':

    distributions_logger = Logs(logger_name='distributions_logs', log_file_name='distributions_logs_6_6_17')
    result_logger = Logs(logger_name='result_logs', log_file_name='result_logs_6_6_17')
    for i in range(2, 3):
        all_distributions = {}
        number_UAVs = 8 + 2 * i
        number_gaols = 4 + 1 * i
        all_number_agent_attack_mean = []
        r_env, r_number_agent, r_load_file, r_need_render = run_mpe(
            load_file='../distribution_policy/run_distribution_e3_syn_para_double_random_6_6_17',
            run_file='simple_mul_target.py', number_agent=number_UAVs,
            number_landmark=number_gaols,
            need_render=True)
        print(f"环境初始化成功,number_UAVs:{number_UAVs},number_gaols:{number_gaols}")
        for j in range(1):
            distributions = policy_run(r_env, r_number_agent, r_load_file, r_need_render)
            all_goal_distributions = {}
            for distribution in distributions:
                for key_index, key_item in enumerate(list(distribution.keys())):
                    if len(distribution.get(key_item)) > 3:
                        continue
                    else:
                        if all_goal_distributions.get(key_item, 0) == 0:
                            all_goal_distributions[key_item] = distribution.get(key_item)
                        else:
                            for agent_index in distribution.get(key_item):
                                all_goal_distributions[key_item].append(agent_index)
            # 统计这个轮次的平均值以及每个回合的分布情况
            str_logs = f"当前轮数{j}_无人机数目{number_UAVs}_目标数目{number_gaols}\n"
            number_agent_attack_mean = 0
            for key, values in all_goal_distributions.items():
                str_logs += str(key) + ':' + str(values) + '\n'
                number_agent_attack_mean += len(values)
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
