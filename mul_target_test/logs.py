# -*- coding: utf-8 -*-
# @Time : 2022/6/10 下午1:48
# @Author :  wangshulei
# @FileName: logs.py
# @Software: PyCharm
import logging
import os


class Logs:
    def __init__(self, logger_name, log_file_name):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        if not os.path.exists('/home/pipiw/PycharmProjects/rl_swarm_distirbution/mul_target_test/rl_logs'):
            os.makedirs('/home/pipiw/PycharmProjects/rl_swarm_distirbution/mul_target_test/rl_logs')
        self.log_handler = logging.FileHandler('rl_logs/' + log_file_name + '.log', 'a', encoding='utf-8')
        self.log_handler.setLevel(logging.DEBUG)
        # formatter = logging.Formatter(
        #     '%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')
        formatter = logging.Formatter(
            '%(filename)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        self.logger.addHandler(self.log_handler)

    def add_logs(self, str_logs):
        self.logger.debug(str_logs)
