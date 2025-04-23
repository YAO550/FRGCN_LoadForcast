# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import z_score

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        # 确保stats是包含'mean'和'std'键的字典
        self.mean = stats['mean']  # 形状需与数据匹配 (如[n_route, 1])
        self.std = stats['std']  # 例如: mean.shape = (11, 1)

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}  # 返回字典
    #
    # def __init__(self, data, stats):
    #     self.__data = data
    #     self.mean = stats['mean']
    #     self.std = stats['std']
    #
    # def get_data(self, type):
    #     return self.__data[type]
    #
    # def get_stats(self):
    #     return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    '''
    按时间顺序生成连续的时间窗口序列
    '''
    n_slot = day_slot - n_frame + 1
    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))

    for i in range(len_seq):
        for j in range(n_slot):
            # 按时间顺序切割数据
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def data_gen(file_path, data_config, n_route, n_frame=96, day_slot=288):
# def data_gen(file_path, data_config, n_route, n_frame=96, day_slot=1440):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = data_config
    data_seq = pd.read_csv(file_path, header=None).values  # 假设数据已按时间排序

    # 按时间顺序划分数据集
    seq_train = data_seq[0: n_train * day_slot]  # 前 n_train 天
    seq_val = data_seq[n_train * day_slot: (n_train + n_val) * day_slot]  # 中间 n_val 天
    seq_test = data_seq[(n_train + n_val) * day_slot:]  # 最后 n_test 天

    # # 生成序列（调用修改后的 seq_gen）
    # seq_train = seq_gen(n_train, seq_train, 0, n_frame, n_route, day_slot)
    # seq_val = seq_gen(n_val, seq_val, 0, n_frame, n_route, day_slot)
    # seq_test = seq_gen(n_test, seq_test, 0, n_frame, n_route, day_slot)

    # 计算全局偏移量
    offset_train = 0
    offset_val = n_train
    offset_test = n_train + n_val

    # 生成训练集、验证集、测试集序列，传入正确的全局offset
    seq_train = seq_gen(n_train, data_seq, offset_train, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, offset_val, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, offset_test, n_frame, n_route, day_slot)

    # 标准化处理
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    return Dataset({'train': x_train, 'val': x_val, 'test': x_test}, x_stats)


def gen_batch(inputs, batch_size, dynamic_batch=False,shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''

    len_inputs = len(inputs)
    # for start_idx in range(0, len_inputs, batch_size):
    #     end_idx = start_idx + batch_size
    #     print(f'Loading batch: {start_idx} to {end_idx}')  # 打印批次范围

    len_inputs = len(inputs)
    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:  # 禁用动态批次调整
                end_idx = len_inputs
            else:
                break
        # 按顺序切片，不进行随机化
        slide = slice(start_idx, end_idx)
        yield inputs[slide]
