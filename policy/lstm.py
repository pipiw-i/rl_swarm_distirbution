# -*- coding: utf-8 -*-
# @Time : 2022/5/20 下午3:59
# @Author :  wangshulei
# @FileName: lstm.py
# @Software: PyCharm
import os
import tensorflow as tf

# 按需分配显存
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class Lstm:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.lstm_model = self.lstm_init()

    def lstm_init(self):
        input_obs = tf.keras.Input(shape=(None,))
        INPUT = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_obs)
        lstm = tf.keras.layers.LSTM(self.output_dim)(INPUT)
        model = tf.keras.Model(inputs=input_obs,
                               outputs=lstm,
                               name='lstm')
        return model

    def save_lstm(self, save_file):
        print('... saving lstm models ...')
        if not os.path.exists(save_file + '/maddpg_model/lstm'):
            os.makedirs(save_file + '/maddpg_model/lstm')
        self.lstm_model.save(save_file + '/maddpg_model/lstm/lstm.h5')

    def load_lstm(self, save_file):
        print('... loading lstm models ...')
        self.lstm_model = tf.keras.models.load_model(save_file + '/maddpg_model/lstm/lstm.h5')
