# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午1:48
# @Author :  wangshulei
# @FileName: calculate_move.py
# @Software: PyCharm
"""
计算动作的文件，通过网络输出的动作，输出攻击工作的智能体，能够直线的冲往目标，这里是在测试的时候编写
"""


class calculate_move:
    def __init__(self, move_action_dim, agent_number):
        self.move_action_dim = move_action_dim
        self.agent_number = agent_number

    def calculate(self, goal_position):
        # TODO:输入目标的位置，计算各个智能体的动作，采取攻击的单位直线冲刺，其余则执行其他的动作
        pass
