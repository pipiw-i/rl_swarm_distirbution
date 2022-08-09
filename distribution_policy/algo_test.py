# -*- coding: utf-8 -*-
# @Time : 2022/6/3 下午10:18
# @Author :  wangshulei
# @FileName: algo_test.py
# @Software: PyCharm
"""
训练智能体文件
"""
import copy

import numpy as np

from distribution_policy.distri_policy import maddpg_policy
from distribution_policy.mpe_env import mpe_env
from distribution_policy.mpe_env_s import mpe_env as mpe_env_s_reward

SEED = 65535
ACTION_SPAN = 0.5


def run_mpe(save_file, actor_learning_rate, critic_learning_rate,
            run_file='simple_search_team', n_game=6000,
            s_reward=False):
    if not s_reward:
        env = mpe_env(run_file, seed=SEED, test=True)
    else:
        env = mpe_env_s_reward(run_file, seed=SEED, test=True)
    action_dim = env.get_action_space()
    obs_dim = env.get_obs_space()
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = maddpg_policy(obs_dim=obs_dim, action_dim=action_dim,
                                  agent_number=agent_number,
                                  actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                                  action_span=ACTION_SPAN, soft_tau=1e-2, log_dir=save_file + '/results')
    maddpg_agents.load_test_models(save_file='run_distribution_e3_syn_para_double_random_6_6_10',
                                   episode=6000)
    n_epi = 50
    i_epi = 0
    attack_number_test = []
    while i_epi < n_epi:
        obs_n = env.mpe_env.reset()
        attack_number_list = []
        n_game = 10
        n_game_calculate = 0
        old_attack_number_index_list = []
        new_attack_number_index_list = []
        attack_number = 0
        for t in range(n_game):
            # 决策攻击的无人机
            env.mpe_env.render()
            action_n = maddpg_agents.get_all_test_action(obs_n, old_attack_number_index_list, show=False)
            action_real = []
            for action in action_n:
                action_real.append(action)
            new_obs_n, reward_n, done_n, info_n, attack_number, attack_number_index_list = env.mpe_env.step(action_real)
            if not old_attack_number_index_list:
                # 如果是空，那么就设置为old_attack_number_index_list
                old_attack_number_index_list = copy.deepcopy(attack_number_index_list)
            else:
                # 如果不是空，那么只选择上次选择过的无人机编号
                for attack_index in attack_number_index_list:
                    if attack_index in old_attack_number_index_list:
                        new_attack_number_index_list.append(copy.deepcopy(attack_index))
                # 说明下一次的进行更新，没有无人机进行攻击了,那么不进行更新，仍保留上次的攻击无人机
                if len(new_attack_number_index_list) != 0:
                    old_attack_number_index_list = copy.deepcopy(new_attack_number_index_list)
                else:
                    old_attack_number_index_list = copy.deepcopy(old_attack_number_index_list)
                new_attack_number_index_list.clear()
            attack_number_list.append(copy.deepcopy(len(old_attack_number_index_list)))
            obs_n = new_obs_n
            if 0 < len(old_attack_number_index_list) <= 3:  # 如果此时以及选出合适数目的无人机，那么结束，输出攻击的无人机数目
                break
        print(f"attack_number_change_list is {attack_number_list},"
              f"final_attack_number is {len(old_attack_number_index_list)},"
              f"final_attack_agent_index is {old_attack_number_index_list}")
        env.set_move()
        for t in range(30 - n_game):
            # 决策攻击的无人机
            env.mpe_env.render()
            action_n = maddpg_agents.get_all_test_action(obs_n, old_attack_number_index_list, show=True)
            action_real = []
            for action in action_n:
                action_real.append(action)
            new_obs_n, _, _, _, _, _ = env.mpe_env.step(action_real)
            obs_n = new_obs_n

        i_epi += 1
        attack_number_test.append(copy.deepcopy(len(old_attack_number_index_list)))
        old_attack_number_index_list.clear()
    print(f"attack_number_test is {attack_number_test}")
    print(f"attack_number_test_mean = {np.mean(attack_number_test)}")


if __name__ == '__main__':
    run_mpe('run_distribution_e3_synchronization_parameters_test',
            1e-3, 1e-3, n_game=6000, run_file='simple_distribution', s_reward=True)
