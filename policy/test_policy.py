# -*- coding: utf-8 -*-
# @Time : 2022/5/20 下午5:02
# @Author :  wangshulei
# @FileName: test_policy.py
# @Software: PyCharm
from RL_algorithm_package.rddpg.mpe.environment import MultiAgentEnv
import RL_algorithm_package.rddpg.mpe.scenarios as scenarios
from RL_algorithm_package.rddpg.script.utils import space_n_to_shape_n
from RL_algorithm_package.rddpg.policy.actor_critic_net import critic, actor
from RL_algorithm_package.rddpg.policy.convert_obs import convert
import numpy as np
import tensorflow as tf


class mpe_env:
    def __init__(self,
                 mpe_env_name,
                 seed):
        self.mpe_env_name = mpe_env_name
        self.seed = seed
        self.mpe_env = self.env_init()

    def env_init(self):
        scenario = scenarios.load(self.mpe_env_name + '.py').Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            info_callback=None,
                            shared_viewer=True)
        env.seed(self.seed)
        return env

    def get_space(self):
        obs_shape_n = space_n_to_shape_n(self.mpe_env.observation_space)
        act_shape_n = space_n_to_shape_n(self.mpe_env.action_space)
        return obs_shape_n[0][0], act_shape_n[0][0]

    def get_agent_number(self):
        return self.mpe_env.n


obs_dim = 32
if __name__ == "__main__":
    world = mpe_env('simple_search', 65535)
    obs_1 = world.mpe_env.reset()
    obs = np.array([0, 0, 1.25, 1.25])
    world.mpe_env.render()
    policy_actor = actor(obs_dim, 5, 1, 1e-2, 0, True, 'actor', 0.5)
    convert_obs_fun = convert(obs_dim)
    for n_games in range(100):
        convert_obs_fun.lstm_model.load_lstm('lstm')
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        obs = convert_obs_fun.convert_obs(obs)
        action_n = policy_actor.actor(obs)
        # new_obs_n, reward_n, done_n, info_n = world.mpe_env.step(action_n)
        # print(f"new_obs_n is {new_obs_n},reward_n is {reward_n},done_n is {done_n}")
        # obs = new_obs_n
        # world.mpe_env.render()
        convert_obs_fun.lstm_model.save_lstm('lstm')
    print("over")
