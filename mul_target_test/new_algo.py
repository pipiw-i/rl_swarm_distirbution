# -*- coding: utf-8 -*-
# @Time : 2022/6/7 下午13:18
# @Author :  wangshulei
# @FileName: algo_mul_target_test.py
# @Software: PyCharm
import copy

from tqdm import tqdm
from mul_target_test.total_policy import AgentPolicy
from mul_target_test.logs import Logs
from mul_target_test.mpe_mul_target_env import mpe_env

SEED = 65535
ACTION_SPAN = 0.5


# 所有智能体的策略
def policy_run(env, number_agent, landmark_number, load_file, need_render, grouping_ratio):
    agent_policy = AgentPolicy(agent_number=number_agent,
                               landmark_number=landmark_number,
                               load_file=load_file,
                               grouping_ratio=grouping_ratio).init_policy()
    obs_n = env.mpe_env.reset()
    landmark_attack_agent_index_dic_list = []
    reassign_goals = False  # 重分配标志位
    gaols_finish = False  # 目标完成
    agent_finish = False  # 智能体完成
    for t in tqdm(range(500)):
        # 决策攻击的无人机
        if need_render:
            env.mpe_env.render()
        action_n = []
        # 根据环境的观测，得到动作
        for agent_index in range(number_agent):
            action_n.append(copy.deepcopy(agent_policy[agent_index](obs=obs_n[agent_index],
                                                                    time_step=t,
                                                                    agent_index=agent_index)))
        # 统计部分，根据action统计每个目标所攻击的无人机编号
        landmark_attack_agent_index_dic = {}
        real_action = []
        # 根据action,输入给环境，使其发生变化
        for action in action_n:
            # 说明该部分是由rl产生的
            if len(action) == 3:
                # 统计所有攻击决策的无人机
                rl_act, obs_landmarks_feature, agent_index_number = action
                real_action.append(copy.deepcopy(rl_act))
            else:
                real_action.append(copy.deepcopy(action))
        # 是否重新分配也是观测参数的一部分
        new_obs_n = env.mpe_env.step([real_action, reassign_goals,
                                      landmark_attack_agent_index_dic_list, grouping_ratio])
        # 通过观测值来完成迭代算法。之前采用的是全局信息来做的，这里更改为通过观测值的方式来获取
        obs_n = new_obs_n
        if len(landmark_attack_agent_index_dic_list) == number_gaols:
            gaols_finish = True
        destroyed_number_count = 0
        for a, _, _, _, _, _, _, _, _, _, decision_finish in new_obs_n:
            if a.is_destroyed:
                destroyed_number_count += 1
            # decision_finish[0] = True 时，说明此时已经完成了决策计算
            # 决策的迭代过程都在智能体内部进行
            elif decision_finish[0]:
                # 在观测值中的统计智能体以及目标信息。程序运行到这里说明此时已经完成了迭代过程
                a_agent = decision_finish[1]
                a_landmark = decision_finish[2]
                if landmark_attack_agent_index_dic.get(a_landmark[0], 0) == 0:
                    landmark_attack_agent_index_dic[a_landmark[0]] = [a_agent.index_number]
                else:
                    landmark_attack_agent_index_dic[a_landmark[0]].append(a_agent.index_number)
            else:
                continue
        # 决策完成之后，统计输出
        attack_landmarks = list(landmark_attack_agent_index_dic.keys())
        for attack_landmark in attack_landmarks:
            landmark_exist = False
            attack_this_landmark_agent_index = landmark_attack_agent_index_dic.get(attack_landmark)
            for attack_dict in landmark_attack_agent_index_dic_list:
                if list(attack_dict.keys())[0] == attack_landmark:
                    landmark_exist = True
                    for a_index in attack_this_landmark_agent_index:
                        attack_dict[attack_landmark].append(a_index)
            if not landmark_exist:
                landmark_attack_agent_index_dic_list.append({attack_landmark: attack_this_landmark_agent_index})
        if destroyed_number_count == number_agent:
            agent_finish = True
        if agent_finish:
            # 提前退出
            pass
        elif gaols_finish and not agent_finish:
            # print("现在进行重分配")
            # 假如不知道全部的目标个数，那么重分配的启动可以由时间参数来启动
            # 比如时间参数设置为当大于500时，启动重启程序
            reassign_goals = True

    return landmark_attack_agent_index_dic_list


def run_mpe(load_file, run_file='simple_search_team', number_agent=8,
            number_landmark=1, need_search_agent=False, need_render=False, grouping_ratio=0.5):
    env = mpe_env(run_file, seed=SEED, number_agent=number_agent,
                  number_landmark=number_landmark,
                  need_search_agent=need_search_agent,
                  grouping_ratio=grouping_ratio)
    return env, number_agent, load_file, need_render


if __name__ == '__main__':
    group_ratio = 0  # 分成两组的比例
    search_agent = True  # 是否增加侦查智能体
    distributions_logger = Logs(logger_name='distributions_logs', log_file_name='distributions_logs_6_6_17')
    result_logger = Logs(logger_name='result_logs', log_file_name='result_logs_6_6_17')
    for i in range(2, 3):
        all_distributions = {}
        number_UAVs = 8 + 8 * i  # 设置智能体个数
        number_gaols = 4 + 4 * i  # 设置目标个数
        all_number_agent_attack_mean = []  # 统计内容
        r_env, r_number_agent, r_load_file, r_need_render = run_mpe(
            load_file='../distribution_policy/run_distribution_e3_syn_para_double_random_6_6_17',
            run_file='simple_mul_target.py', number_agent=number_UAVs,
            number_landmark=number_gaols,
            need_search_agent=search_agent,
            need_render=True,
            grouping_ratio=group_ratio)
        print(f"环境初始化成功,number_UAVs:{number_UAVs},number_gaols:{number_gaols}")
        for j in range(3):
            # 执行策略，得到分配结果
            distributions = policy_run(r_env, r_number_agent, number_gaols,
                                       r_load_file, r_need_render, group_ratio)
            # 以下是统计内容
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
        # 这里是50次的平均值
        str_logs = f"result_{number_UAVs}_{number_gaols}_50_times\n"
        mean_50 = 0
        for times, one_number_agent_attack_mean in enumerate(all_number_agent_attack_mean):
            str_logs += f"当前轮数{times},平均攻击次数{one_number_agent_attack_mean}\n"
            mean_50 += one_number_agent_attack_mean
        mean_50 /= 50
        str_logs += f"在无人机数目{number_UAVs}目标数目{number_gaols}下，平均攻击次数为{mean_50}\n"
        result_logger.add_logs(str_logs)
    print("程序运行完毕！")
