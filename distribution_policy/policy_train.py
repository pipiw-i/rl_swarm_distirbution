# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午2:45
# @Author :  wangshulei
# @FileName: policy_train.py
# @Software: PyCharm
"""
强化学习policy训练文件，run 这个文件开始训练
"""
import copy
import datetime
import os

import numpy as np
from matplotlib import pyplot as plt

from distribution_policy.distri_exp import SharedExp
from distribution_policy.distri_policy import maddpg_policy
from distribution_policy.mpe_env import mpe_env
from distrzhuibution_policy.mpe_env_s import mpe_env as mpe_env_s_reward

SEED = 65535
ACTION_SPAN = 0.5


def run_mpe(save_file, actor_learning_rate, critic_learning_rate,
            run_file='simple_search_team', n_game=6000,
            s_reward=False):
    if not s_reward:
        env = mpe_env(run_file, seed=SEED)
    else:
        env = mpe_env_s_reward(run_file, seed=SEED)
    action_dim = env.get_action_space()
    obs_dim = env.get_obs_space()  # 7
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = maddpg_policy(obs_dim=obs_dim, action_dim=action_dim,
                                  agent_number=agent_number,
                                  actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                                  action_span=ACTION_SPAN, soft_tau=1e-2, log_dir=save_file + '/results')
    maddpg_agents.synchronization_parameters()
    # maddpg_agents.load_models('run_distribution_s_e3_synchronization_parameters_double_random_UAVs', 1000)
    # 经验池初始化
    all_agent_exp = []
    for agent_index in range(agent_number):
        exp = SharedExp(exp_size=51200, batch_size=64, obs_dim=obs_dim, action_dim=action_dim,
                        r_dim=1, done_dim=1, agent_number=agent_number, agent_index=agent_index)
        all_agent_exp.append(exp)
    score = []
    avg_score = []
    explore_span = 0.2 * ACTION_SPAN
    # 暖机，得到足够用来学习的经验
    while not all_agent_exp[0].can_learn():
        obs_n = env.mpe_env.reset()
        for t in range(30):
            env.mpe_env.render()
            action_n = maddpg_agents.get_all_action(obs_n, explore_span)
            new_obs_n, reward_n, done_n, info_n, attack_number, _ = env.mpe_env.step(action_n)
            # 存入经验
            for agent_index in range(agent_number):
                all_agent_exp[agent_index].exp_store(obs_n, action_n, reward_n, new_obs_n, done_n)
            obs_n = new_obs_n
    learning_step = 0
    for i_episode in range(n_game):
        obs_n = env.mpe_env.reset()
        score_one_episode = 0
        attack_number_list = []
        for t in range(30):
            learning_step += 1
            if learning_step % 10 == 0:
                maddpg_agents.synchronization_parameters()
            env.mpe_env.render()
            # 探索的幅度随机训练的进行逐渐减小
            if n_game > 3000:
                if i_episode >= (n_game - 3000) and ((i_episode - 3000) % 500 == 0):
                    explore_span = 0.2 * ((n_game - i_episode) / 3000) * ACTION_SPAN
                    print(f"change the explore_span {explore_span}")
            action_n = maddpg_agents.get_all_action(obs_n, explore_span)
            new_obs_n, reward_n, done_n, info_n, attack_number, _ = env.mpe_env.step(action_n)
            attack_number_list.append(copy.deepcopy(attack_number))
            exp_index_n = []
            for agent_index in range(agent_number):
                all_agent_exp[agent_index].exp_store(obs_n, action_n, reward_n, new_obs_n, done_n)
            obs_n = new_obs_n
            # 全都采用相同时刻的经验进行而学习
            index, _ = all_agent_exp[0].sample()
            for agent_index in range(agent_number):
                exp_index_n.append(index)
            maddpg_agents.update(all_agent_exp, exp_index_n)
            score_one_episode += (reward_n[0] + reward_n[1] + reward_n[2] + reward_n[3]) / 4
        maddpg_agents.logs(score_one_episode, learning_step)
        if (i_episode + 1) % 1000 == 0:
            if not os.path.exists(save_file + '/rmaddpg_img'):
                os.makedirs(save_file + '/rmaddpg_img')
            plt.plot(score)  # 绘制波形
            plt.plot(avg_score)  # 绘制波形
            plt.savefig(save_file + f"/rmaddpg_img/rmaddpg_score:{i_episode + 1}.png")
            maddpg_agents.save_models(save_file, i_episode + 1)
        score.append(score_one_episode)
        avg = np.mean(score[-100:])
        avg_score.append(avg)
        print(f"i_episode is {i_episode},score_one_episode is {score_one_episode},avg_score is {avg},"
              f"attack_number_list is {attack_number_list}")
    env.mpe_env.close()
    plt.close()


if __name__ == '__main__':
    # time_now = datetime.datetime.now()
    # print("当前的日期和时间是 %s" % time_now)
    # run_mpe(f'run_distribution_e3_syn_para_double_random_{time_now.month}_{time_now.day}_{time_now.hour}',
    #         1e-3, 1e-3, n_game=6000, run_file='simple_distribution', s_reward=True)
    time_now = datetime.datetime.now()
    print("当前的日期和时间是 %s" % time_now)
    run_mpe(f'run_distribution_s_e3_syn_para_double_random_{time_now.month}_{time_now.day}_{time_now.hour}',
            1e-3, 1e-3, n_game=6000, run_file='simple_distribution', s_reward=False)
