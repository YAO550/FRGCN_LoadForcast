from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    timestamps = []  # 新增时间戳收集
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        timestamps.append(df.index[t])  # 收集原始时间戳
    
    # 转换为numpy datetime类型
    # Convert lists to numpy arrays
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    timestamps = np.array(timestamps, dtype='datetime64[m]')
    
    return x, y, timestamps  # Now returns numpy arrays instead of lists

def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, timestamps = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )
    
    # 修改后的顺序分割逻辑
    num_samples = x.shape[0]
    
    # 直接取前70%作为训练集
    num_train = int(num_samples * 0.7)
    # 中间15%作为验证集
    num_val = int(num_samples * 0.15)
    # 最后15%作为测试集
    num_test = num_samples - num_train - num_val
    
    # 显式定义数据集分割
    x_train, x_val, x_test = x[:num_train], x[num_train:num_train+num_val], x[-num_test:]
    y_train, y_val, y_test = y[:num_train], y[num_train:num_train+num_val], y[-num_test:]
    
    # 时间戳分割保持相同比例
    timestamps_train = timestamps[:num_train]
    timestamps_val = timestamps[num_train:num_train+num_val]
    timestamps_test = timestamps[-num_test:]
    
    # 保存时添加时间戳
    for cat in ["train", "val", "test"]:
        _x, _y, _timestamps = locals()[f"x_{cat}"], locals()[f"y_{cat}"], locals()[f"timestamps_{cat}"]
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            timestamps=_timestamps,  # 新增时间戳字段
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/NewYork", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/NewYork/output.h5", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
