from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from os.path import join as pjoin
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pandas as pd
import os

global_step = 0
n_route = 11


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    pred_collector = []
    actual_collector = []

    if seq.shape[1] < n_his + n_pred:
        raise ValueError(f"输入数据时间步不足: 需要 {n_his + n_pred} 步，实际 {seq.shape[1]} 步")

    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch, shuffle=False):
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        actuals = np.copy(i[:, n_his:, :, :])

        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred, feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)

        pred_collector.append(np.array(step_list))
        actual_collector.append(actuals.transpose(1, 0, 2, 3))

    pred_array = np.concatenate(pred_collector, axis=1)
    actual_array = np.concatenate(actual_collector, axis=1)
    return pred_array[step_idx], pred_array.shape[1], pred_array, actual_array


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: n_pred "{n_pred}" exceeds limit.')

    y_val, len_val, full_pred_val, full_actual_val = multi_pred(
        sess, pred, x_val, batch_size, n_his, n_pred, step_idx
    )

    y_test, len_test, full_pred_test, full_actual_test = multi_pred(
        sess, pred, x_test, batch_size, n_his, n_pred, step_idx
    )

    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)
    chks = evl_val < min_va_val
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        evl_pred = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
        min_val = evl_pred

    return min_va_val, min_val, [full_pred_val, full_pred_test], [full_actual_val, full_actual_test]


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    global global_step
    start_time = time.time()

    load_path = os.path.abspath(load_path or './output/models/')
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"模型目录不存在: {load_path}")

    ckpt = tf.train.get_checkpoint_state(load_path)
    if not ckpt or not ckpt.model_checkpoint_path:
        raise ValueError(f"无有效检查点: {load_path}")
    model_path = ckpt.model_checkpoint_path

    x_test = inputs.get_data('test')
    if x_test.shape[1] < n_his + n_pred:
        raise ValueError(f"测试数据时间步不足: 需要 {n_his + n_pred} 步，实际 {x_test.shape[1]} 步")

    test_graph = tf.Graph()
    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, model_path)
        print(f'>> 加载模型: {model_path}')

        pred = test_graph.get_collection('y_pred')

        # 获取完整预测结果
        _, _, full_preds, full_actuals = multi_pred(
            test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx=np.arange(n_pred),
            dynamic_batch=False
        )

        # 反归一化处理
        x_stats = inputs.get_stats()
        data_mean, data_std = x_stats['mean'], x_stats['std']

        # 调整维度顺序 [n_pred, batch, n_route, 1] => [batch, n_pred, n_route, 1]
        test_preds = full_preds.transpose(1, 0, 2, 3)
        test_actuals = full_actuals.transpose(1, 0, 2, 3)

        # 反归一化
        test_preds_denorm = test_preds * data_std + data_mean
        test_actuals_denorm = test_actuals * data_std + data_mean

        # 生成结果记录
        results = []
        for i in range(test_preds_denorm.shape[0]):  # 遍历每个样本
            for t in range(test_preds_denorm.shape[1]):  # 遍历每个时间步
                record = {'组号': global_step + 1, '时间步': t + 1}
                for route in range(n_route):
                    pred_val = test_preds_denorm[i, t, route, 0]
                    actual_val = test_actuals_denorm[i, t, route, 0]
                    rmse = np.sqrt((actual_val - pred_val) ** 2)
                    record[f'预测值（节点{route + 1}）'] = pred_val
                    record[f'真实值（节点{route + 1}）'] = actual_val
                    record[f'RMSE（节点{route + 1}）'] = rmse
                results.append(record)
            global_step += 1

        # 创建并保存DataFrame
        columns = ['组号', '时间步'] + [f'{t}（节点{r + 1}）' for r in range(n_route) for t in ['预测值', '真实值', 'RMSE']]
        df = pd.DataFrame(results, columns=columns)
        output_dir = './output/results'
        os.makedirs(output_dir, exist_ok=True)
        excel_path = os.path.join(output_dir, '多组预测结果.xlsx')
        df.to_excel(excel_path, index=False)
        print(f'预测结果已保存至: {excel_path}')

        # 指标计算
        if inf_mode == 'sep':
            tmp_idx = [n_pred - 1]
        elif inf_mode == 'merge':
            tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'无效模式: {inf_mode}')

        y_true = x_test[0:full_actuals.shape[1], n_his:n_his + n_pred, :, :]
        y_pred_transposed = full_preds.transpose(1, 0, 2, 3)

        y_true_reshaped = y_true.reshape(-1, y_true.shape[2], 1)
        y_pred_reshaped = y_pred_transposed.reshape(-1, y_pred_transposed.shape[2], 1)

        evl = evaluation(y_true_reshaped, y_pred_reshaped, x_stats)

        if evl.size == 0:
            raise ValueError("评估结果为空，请检查输入数据的形状。")

        for ix in tmp_idx:
            if ix >= len(evl):
                continue
            te = evl[ix - 2:ix + 1] if (ix - 2 >= 0 and ix + 1 <= len(evl)) else evl
            print(f'时间步 {ix + 1}: MAPE {te[0]:7.3%}; MAE {te[1]:4.3f}; RMSE {te[2]:6.3f}')
        print(f'总测试时间: {time.time() - start_time:.2f}秒')

    print('测试完成!')