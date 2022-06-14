# -*- coding: utf-8 -*-
# @Time : 2022/5/20 下午5:30
# @Author :  wangshulei
# @FileName: run.py
# @Software: PyCharm
from matplotlib import pyplot as plt
import os
import numpy as np
from RL_algorithm_package.rddpg.policy.policy import maddpg_policy
from RL_algorithm_package.rddpg.script.mpe_env import mpe_env
from RL_algorithm_package.rddpg.script.mpe_s_reward_env import mpe_env as mpe_env_s_reward
from RL_algorithm_package.rddpg.script.lstm_exp import SharedExp
from RL_algorithm_package.rddpg.policy.convert_obs import convert

SEED = 65535
ACTION_SPAN = 0.5
obs_dim = 8  # lstm 的输出个数


def run_mpe(save_file, actor_learning_rate, critic_learning_rate,
            run_file='simple_search_team', n_game=6000,
            s_reward=False):
    if not s_reward:
        env = mpe_env(run_file, seed=SEED)
    else:
        env = mpe_env_s_reward(run_file, seed=SEED)
    convert_f = convert(obs_dim)
    convert_f.lstm_model.save_lstm(save_file)
    action_dim = env.get_action_space()
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = maddpg_policy(obs_dim=obs_dim, action_dim=action_dim,
                                  agent_number=agent_number,
                                  actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                                  action_span=ACTION_SPAN, soft_tau=1e-2, log_dir=save_file + '/results')
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
        con_obs_n = []  # 经过lstm转换过的观测值
        for obs in obs_n:
            con_obs_n.append(convert_f.convert_obs(obs)[0])
        for t in range(50):
            env.mpe_env.render()
            action_n = maddpg_agents.get_all_action(con_obs_n, explore_span)
            # action_n = [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])]
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(action_n)
            new_con_obs_n = []  # 经过lstm转换过的观测值
            for new_obs in new_obs_n:
                new_con_obs_n.append(convert_f.convert_obs(new_obs)[0])
            # 存入经验
            for agent_index in range(agent_number):
                all_agent_exp[agent_index].exp_store(con_obs_n, action_n, reward_n, new_con_obs_n, done_n)
            con_obs_n = new_con_obs_n

    for i_episode in range(n_game):
        obs_n = env.mpe_env.reset()
        con_obs_n = []  # 经过lstm转换过的观测值
        for obs in obs_n:
            con_obs_n.append(convert_f.convert_obs(obs)[0])
        score_one_episode = 0
        for t in range(50):
            env.mpe_env.render()
            # 探索的幅度随机训练的进行逐渐减小
            if i_episode >= (n_game - 3000) and ((i_episode - 3000) % 500 == 0):
                explore_span = 0.2 * ((n_game - i_episode) / 3000) * ACTION_SPAN
                print(f"change the explore_span {explore_span}")
            action_n = maddpg_agents.get_all_action(con_obs_n, explore_span)
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(action_n)
            # print(f"reward_n is {reward_n}")
            new_con_obs_n = []  # 经过lstm转换过的观测值
            for new_obs in new_obs_n:
                new_con_obs_n.append(convert_f.convert_obs(new_obs)[0])
            exp_index_n = []
            for agent_index in range(agent_number):
                all_agent_exp[agent_index].exp_store(con_obs_n, action_n, reward_n, new_con_obs_n, done_n)
            # 全都采用相同时刻的经验进行而学习
            index, _ = all_agent_exp[0].sample()
            for agent_index in range(agent_number):
                exp_index_n.append(index)
            maddpg_agents.update(all_agent_exp, exp_index_n)
            # print(f"reward_n is {reward_n}")
            score_one_episode += (reward_n[0] + reward_n[1])/2
            con_obs_n = new_con_obs_n
        maddpg_agents.logs(score_one_episode, 50 * (i_episode + 1))
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
        print(f"i_episode is {i_episode},score_one_episode is {score_one_episode},avg_score is {avg}")
    env.mpe_env.close()
    plt.close()


if __name__ == '__main__':
    # run_mpe('run_random_object_T_REWARD', 1e-3, 1e-3, n_game=6000, run_file='simple_search_team', s_reward=False)
    # run_mpe('run_random_object_S_REWARD', 5e-4, 5e-4, n_game=6000, run_file='simple_search_team', s_reward=True)
    run_mpe('run_random_object_full_obs_3_1e3', 1e-3, 1e-3, n_game=6000, run_file='simple_search_team', s_reward=True)
    run_mpe('run_random_object_full_obs_3_1e3_T', 1e-3, 1e-3, n_game=6000, run_file='simple_search_team', s_reward=False)
    run_mpe('run_random_object_full_obs_3_1e3', 5e-4, 5e-4, n_game=6000, run_file='simple_search_team', s_reward=True)
