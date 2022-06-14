# -*- coding: utf-8 -*-
# @Time : 2022/5/21 上午10:02
# @Author :  wangshulei
# @FileName: test.py
# @Software: PyCharm
import time

import numpy as np

from policy.convert_obs import convert
from policy.policy import maddpg_policy
from script.mpe_env import mpe_env

SEED = 65535
ACTION_SPAN = 0.5
obs_dim = 8  # lstm 的输出个数


def test_mpe(save_file, actor_learning_rate, critic_learning_rate):
    env = mpe_env('simple_search_team', seed=SEED)
    convert_f = convert(obs_dim)
    convert_f.lstm_model.load_lstm(save_file)
    action_dim = env.get_action_space()
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = maddpg_policy(obs_dim=obs_dim, action_dim=action_dim,
                                  agent_number=agent_number,
                                  actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                                  action_span=ACTION_SPAN, soft_tau=1e-2, log_dir=save_file + '/results')
    # 加载训练好的数据
    maddpg_agents.load_models(save_file, 6000)
    score = []
    avg_score = []
    for i_episode in range(6000):
        obs_n = env.mpe_env.reset()
        con_obs_n = []  # 经过lstm转换过的观测值
        for obs in obs_n:
            con_obs_n.append(convert_f.convert_obs(obs)[0])
        score_one_episode = 0
        for t in range(50):
            env.mpe_env.render()
            time.sleep(0.03)
            action_n = maddpg_agents.get_all_test_action(con_obs_n)
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(action_n)
            new_con_obs_n = []  # 经过lstm转换过的观测值
            for new_obs in new_obs_n:
                new_con_obs_n.append(convert_f.convert_obs(new_obs)[0])
            score_one_episode += reward_n[0]
            print(f"reward_n is {reward_n}")
            con_obs_n = new_con_obs_n
        maddpg_agents.logs(score_one_episode, 20 * (i_episode + 1))
        score.append(score_one_episode)
        avg = np.mean(score[-100:])
        avg_score.append(avg)
        print(f"i_episode is {i_episode},score_one_episode is {score_one_episode},avg_score is {avg}")
    env.mpe_env.close()


if __name__ == '__main__':
    test_mpe('run_random_object_full_obs_1e3_T', 1e-3, 1e-3)
