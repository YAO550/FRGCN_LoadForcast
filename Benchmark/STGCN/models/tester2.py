from data_loader.data_utils import gen_batch

from utils.math_utils import evaluation
from os.path import join as pjoin
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pandas as pd
import os

global_step = 0
c=1
n_route=11
def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    返回完整预测序列和对应真实值（已修复时间步校验）
    '''

    '''
    多步预测函数（返回完整预测序列和对应真实值）
    - sess: TensorFlow会话
    - y_pred: 预测操作符
    - seq: 输入数据序列
    - batch_size: 批次大小
    - n_his: 历史时间步数
    - n_pred: 预测时间步数
    - step_idx: 需要返回的预测步索引
    - dynamic_batch: 是否动态调整批次大小
    '''

    pred_collector = []
    actual_collector = []

    # 确保输入数据足够支持预测步数
    if seq.shape[1] < n_his + n_pred:
        raise ValueError(
            f"输入数据时间步不足: 需要 {n_his + n_pred} 步，实际 {seq.shape[1]} 步"
        )

    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch, shuffle=False):
        test_seq = np.copy(i[:, 0:n_his+1, :, :])
        actuals = np.copy(i[:, n_his:, :, :])  # [batch, n_pred, n_route, 1]

        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred, feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)

        pred_collector.append(np.array(step_list))  # [n_pred, batch, n_route, 1]
        actual_collector.append(actuals.transpose(1, 0, 2, 3))  # [n_pred, batch, n_route, 1]

    pred_array = np.concatenate(pred_collector, axis=1)
    actual_array = np.concatenate(actual_collector, axis=1)
    return pred_array[step_idx], pred_array.shape[1], pred_array, actual_array


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    返回验证集和测试集的完整预测结果
    '''
    '''
    模型推理函数（验证集+测试集）
    - sess: TensorFlow会话
    - pred: 预测操作符
    - inputs: 数据输入对象
    - step_idx: 需要评估的预测步索引
    - min_va_val/min_val: 历史最佳指标值
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: n_pred "{n_pred}" exceeds limit.')

    # 获取验证集预测
    y_val, len_val, full_pred_val, full_actual_val = multi_pred(
        sess, pred, x_val, batch_size, n_his, n_pred, step_idx
    )

    # 获取测试集预测
    y_test, len_test, full_pred_test, full_actual_test = multi_pred(
        sess, pred, x_test, batch_size, n_his, n_pred, step_idx
    )

    # 指标计算
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)
    chks = evl_val < min_va_val
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        evl_pred = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
        min_val = evl_pred

    return min_va_val, min_val, [full_pred_val, full_pred_test], [full_actual_val, full_actual_test]


def group_predict(sess, y_pred, data, batch_size, n_group, n_his, n_pred, n_nodes):
    """
    修正后的分组预测函数，调整输入数据的维度顺序。
    """
    """
    分组预测函数（按时间序列分组预测）
    - data: 输入数据 [total_steps, nodes]
    - n_group: 总组数
    - n_nodes: 节点数量
    """

    if data.ndim == 2:
        data = data[..., np.newaxis]  # [steps, nodes] => [steps, nodes, 1]

    seq = np.expand_dims(data, axis=-1)  # [steps, nodes, 1] => [steps, nodes, 1, 1]

    pred_groups = np.zeros((n_group, n_pred, n_nodes))
    true_groups = np.zeros((n_group, n_pred, n_nodes))

    for g in range(n_group):
        start_idx = g * (n_his + n_pred)
        end_idx = start_idx + n_his + n_pred

        group_data = seq[start_idx:end_idx]  # [n_his + n_pred, nodes, 1, 1]

        # 调整维度顺序为 [batch, time_steps, nodes, features]
        test_seq = group_data[:n_his + 1]  # [n_his + 1, nodes, 1, 1]
        test_seq = test_seq.transpose(1, 0, 2, 3)  # 错误转置，应移除或调整
        # 修正：直接添加批次维度并调整形状
        test_seq = test_seq[np.newaxis]  # [1, n_his+1, nodes, 1, 1]
        # 调整维度顺序为 [1, time_steps, nodes, 1]
        test_seq = test_seq.transpose(0, 2, 1, 3, 4)  # 将时间步移到第二维
        test_seq = test_seq.squeeze(axis=-1)  # 移除最后一个多余的维度


        step_pred = []
        current_seq = np.copy(test_seq)
        for step in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': current_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.squeeze()

            # 更新序列，保持维度正确
            new_frame = np.zeros((1, 1, n_nodes, 1))
            new_frame[0, 0, :, 0] = pred
            current_seq = np.concatenate([current_seq[:, 1:, :, :], new_frame], axis=1)

            step_pred.append(pred)

        pred_groups[g] = np.array(step_pred)
        true_groups[g] = group_data[n_his:, :, 0, 0]

    return pred_groups, true_groups

def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    global global_step,c
    start_time = time.time()

    # 1. 路径检查
    load_path = os.path.abspath(load_path or './output/models/')
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"模型目录不存在: {load_path}")

    # 2. 检查点检查
    ckpt = tf.train.get_checkpoint_state(load_path)
    if not ckpt or not ckpt.model_checkpoint_path:
        raise ValueError(f"无有效检查点: {load_path}")
    model_path = ckpt.model_checkpoint_path

    # 3. 数据校验（关键修复点）
    x_test = inputs.get_data('test')
    if x_test.shape[1] < n_his + n_pred:
        raise ValueError(
            f"测试数据时间步不足: 需要 {n_his + n_pred} 步，实际 {x_test.shape[1]} 步"
        )

    test_graph = tf.Graph()
    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, model_path)
        print(f'>> 加载模型: {model_path}')

        pred = test_graph.get_collection('y_pred')

        # 4. 动态获取实际预测步数（关键修复点）
        _, _, full_preds, full_actuals = multi_pred(
            test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx=np.arange(n_pred),
            dynamic_batch=False
        )
        actual_n_pred = full_preds.shape[0]  # 实际生成的预测步数

        # 反归一化处理
        # 获取预测操作
        pred = test_graph.get_collection('y_pred')

        # 获取原始测试数据
        x_test = inputs.get_data('test')  # [n_test, seq_len, n_route, 1]
        x_stats = inputs.get_stats()
        mean, std = x_stats['mean'], x_stats['std']

        # 数据重组：将测试数据转换为连续序列
        full_data = x_test.squeeze(axis=-1)  # [n_test, seq_len, n_route]
        full_data = full_data.transpose(1, 0, 2)  # [seq_len, n_test, n_route]
        full_data = full_data.reshape(-1, full_data.shape[-1])  # [total_steps, n_route]

        # 计算总组数
        total_steps = full_data.shape[0]
        group_size = n_his + n_pred  # 96
        n_group = total_steps // group_size
        if total_steps % group_size != 0:
            print(f"警告：最后{total_steps % group_size}步数据不足一组将被忽略")

        # 执行分组预测
        pred_all, true_all = group_predict(
            test_sess, pred,
            data=full_data[:n_group * group_size],  # 截取完整组
            batch_size=batch_size,
            n_group=n_group,
            n_his=n_his,
            n_pred=n_pred,
            n_nodes=n_route
        )
        # 反归一化处理
        pred_data = pred_all * std + mean  # shape (n_group, n_pred, n_route)
        true_data = true_all * std + mean  # shape (n_group, n_pred, n_route)

        # 生成结果表格（重构部分）
        results = []
        for g in range(n_group):
            group_pred = pred_data[g]  # (n_pred, n_route)
            group_true = true_data[g]  # (n_pred, n_route)

            # 计算每个节点的RMSE（基于该组所有时间步）
            rmse_per_node = [np.sqrt(np.mean((group_pred[:, node] - group_true[:, node]) ** 2)) for node in range(n_route)]

            # 遍历每个时间步生成记录
            for t in range(n_pred):
                record = {
                    '组号': g + 1,
                }
            for node in range(n_route):
                record[f'预测值（节点{node + 1}）'] = group_pred[t, node]
                record[f'真实值（节点{node + 1}）'] = group_true[t, node]
                record[f'RMSE（节点{node + 1}）'] = rmse_per_node[node]
            results.append(record)

            # 创建列名（添加时间步）
            columns = ['组号']
            for node in range(n_route):
                columns += [
                    f'预测值（节点{node + 1}）',
                    f'真实值（节点{node + 1}）',
                    f'RMSE（节点{node + 1}）'
                ]

            df = pd.DataFrame(results, columns=columns)

            # 保存结果（路径保持不变）
            output_dir = './output/results'
            os.makedirs(output_dir, exist_ok=True)
            excel_path = os.path.join(output_dir, '多组预测结果.xlsx')
            df.to_excel(excel_path, index=False)


        # 6. 指标计算（动态索引控制）
        if inf_mode == 'sep':
            tmp_idx = [actual_n_pred - 1]  # 使用实际预测步数
        elif inf_mode == 'merge':
            tmp_idx = np.arange(3, actual_n_pred + 1, 3) - 1  # 基于实际步数生成
            tmp_idx = tmp_idx[tmp_idx < actual_n_pred]  # 过滤越界索引
        else:
            raise ValueError(f'无效模式: {inf_mode}')

        # 安全评估
        # 提取实际数据并调整形状
        y_true = x_test[0:full_actuals.shape[1], n_his:n_his + actual_n_pred, :, :]
        y_true_reshaped = y_true.reshape(-1, y_true.shape[2], y_true.shape[3])

        # 调整预测数据形状
        y_pred_transposed = full_preds.transpose(1, 0, 2, 3)
        y_pred_reshaped = y_pred_transposed.reshape(-1, y_pred_transposed.shape[2], y_pred_transposed.shape[3])

        # 调用评估函数
        evl = evaluation(y_true_reshaped, y_pred_reshaped, x_stats)


        # 输出结果
        for ix in tmp_idx:
            if ix >= len(evl):
                continue  # 防御性跳过
            te = evl[ix - 2:ix + 1]
            print(f'Step {ix + 1}: MAPE {te[0]:7.3%}; MAE {te[1]:4.3f}; RMSE {te[2]:6.3f}')
        print(f'总测试时间: {time.time() - start_time:.2f}s')

    print('测试完成!')