from data_loader.data_utils import gen_batch
from models.tester import *
from models.base_model import build_model, model_save
from os.path import join as pjoin
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import time
import os

b = -1
a = 0
c = 1


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    global a, b, c
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    os.makedirs(sum_path, exist_ok=True)

    tf.disable_eager_execution()
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)

    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    epoch_step = len_train // batch_size + (1 if len_train % batch_size else 0)

    lr = tf.train.exponential_decay(args.lr, global_steps,
                                    decay_steps=5 * epoch_step,
                                    decay_rate=0.7,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)

    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        x_stats = inputs.get_stats()
        data_mean, data_std = x_stats['mean'], x_stats['std']

        if inf_mode == 'sep':
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        for epoch_i in range(epoch):
            full_train_results = []  # 训练集结果
            full_val_results = []  # 验证集结果
            full_test_results = []  # 测试集结果
            # Training phase
            start_time = time.time()
            train_batches = gen_batch(
                inputs.get_data('train'),
                batch_size,
                dynamic_batch=False,
                shuffle=False
            )

            for step_j, x_batch in enumerate(train_batches):
                summary, _ = sess.run([merged, train_op],
                                      feed_dict={x: x_batch[:, 0:n_his + 1, :, :],
                                                 keep_prob: 1.0})
                writer.add_summary(summary, epoch_i * epoch_step + step_j)

                if step_j % 50 == 0:
                    loss_value = sess.run([train_loss, copy_loss],
                                          feed_dict={x: x_batch[:, 0:n_his + 1, :, :],
                                                     keep_prob: 1.0})
                    print(f'Epoch {epoch_i:2d}, Step {step_j:3d}: '
                          f'[{loss_value[0]:.3f}, {loss_value[1]:.3f}]')

            print(f'Epoch {epoch_i:2d} Training Time {time.time() - start_time:.3f}s')

            # Inference phase
            start_time = time.time()
            min_va_val, min_val, batch_preds, batch_actuals = model_inference(
                sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val
            )

            # 显示指标
            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {epoch_i:2d} Inference Time {time.time() - start_time:.3f}s')

            if (epoch_i + 1) % args.save == 0:
                model_save(sess, global_steps, 'STGCN')

            if epoch_i == epoch - 1:
                # 处理验证集和测试集
                # 解包验证和测试预测结果
                val_preds, test_preds = batch_preds
                val_actuals, test_actuals = batch_actuals

                x_stats = inputs.get_stats()
                data_mean, data_std = x_stats['mean'], x_stats['std']

                # 处理验证集
                # 反归一化处理
                val_preds2 = np.squeeze(val_preds, axis=-1)
                val_actuals2 = np.squeeze(val_actuals, axis=-1)

                val_preds_denorm = val_preds2 * data_std + data_mean
                val_actuals_denorm = val_actuals2 * data_std + data_mean
                for i in range(val_preds_denorm.shape[1]):  # batch维度
                    for t in range(val_preds_denorm.shape[0]):  # 时间步维度
                        c += 1
                        if t % (12 * c) == 0:
                            record = {'组号': global_step, 'epoch': epoch_i}
                            for route in range(n):
                                pred_val = val_preds_denorm[t, i, route]
                                actual_val = val_actuals_denorm[t, i, route]
                                rmse = np.sqrt((actual_val - pred_val) ** 2)
                                record[f'预测值（节点{route + 1}）'] = pred_val
                                record[f'真实值（节点{route + 1}）'] = actual_val
                                record[f'RMSE（节点{route + 1}）'] = rmse
                            full_val_results.append(record)

                # 处理测试集
                test_preds2 = np.squeeze(test_preds, axis=-1)
                test_actuals2 = np.squeeze(test_actuals, axis=-1)

                test_preds_denorm = test_preds2 * data_std + data_mean
                test_actuals_denorm = test_actuals2 * data_std + data_mean
                for i in range(test_preds_denorm.shape[1]):
                    for t in range(test_preds_denorm.shape[0]):
                        if t % (12 * c) == 0:
                            record = {'组号': global_step, 'epoch': epoch_i}
                            for route in range(n):
                                pred_val = test_preds_denorm[t, i, route]
                                actual_val = test_actuals_denorm[t, i, route]
                                rmse = np.sqrt((actual_val - pred_val) ** 2)
                                record[f'预测值（节点{route + 1}）'] = pred_val
                                record[f'真实值（节点{route + 1}）'] = actual_val
                                record[f'RMSE（节点{route + 1}）'] = rmse
                            full_test_results.append(record)

                b += 1
                a = b * 12 + 1  # 每个batch包含12个样本
                # 2处理训练集预测
                # 处理训练集预测
                train_data = inputs.get_data('train')
                train_preds, _, _, _ = multi_pred(sess, pred, train_data, batch_size, n_his, n_pred,
                                                  step_idx=np.arange(n_pred))

                # 转置维度：从 [n_pred, 样本数, n_route, 1] 转为 [样本数, n_pred, n_route, 1]
                train_preds = train_preds.transpose(1, 0, 2, 3)  # 新增转置操作

                # 反归一化
                train_preds_denorm = train_preds * data_std + data_mean  # 形状 (样本数, n_pred, n_route, 1)
                train_actuals_denorm = train_data[:, n_his:n_his + n_pred, :,
                                       :] * data_std + data_mean  # 形状 (样本数, n_pred, n_route, 1)

                # 生成唯一组号并记录
                current_step = sess.run(global_steps)
                for i in range(train_preds_denorm.shape[0]):  # 现在按样本数循环
                    for t in range(n_pred):  # 遍历每个预测时间步
                        if t % (12 * c) == 0:
                            group_id = t + a
                            record = {'组号': group_id, 'epoch': epoch_i}
                            for route in range(n):
                                pred_val = train_preds_denorm[i, t, route, 0]
                                actual_val = train_actuals_denorm[i, t, route, 0]
                                rmse = np.sqrt((actual_val - pred_val) ** 2)
                                record[f'预测值（节点{route + 1}）'] = pred_val
                                record[f'真实值（节点{route + 1}）'] = actual_val
                                record[f'RMSE（节点{route + 1}）'] = rmse
                            full_train_results.append(record)

                # 保存验证集和测试集
                df_val = pd.DataFrame(full_val_results)
                df_val = df_val[
                    ['组号', 'epoch'] + [f'{t}（节点{r + 1}）' for r in range(n) for t in ['预测值', '真实值', 'RMSE']]]
                df_val.to_excel(pjoin(sum_path, f'val_predictions_epoch_{epoch_i}.xlsx'), index=False)

                df_test = pd.DataFrame(full_test_results)
                df_test = df_test[
                    ['组号', 'epoch'] + [f'{t}（节点{r + 1}）' for r in range(n) for t in ['预测值', '真实值', 'RMSE']]]
                df_test.to_excel(pjoin(sum_path, f'test_predictions_epoch_{epoch_i}.xlsx'), index=False)

                df_train = pd.DataFrame(full_train_results)
                df_train = df_train[
                    ['组号', 'epoch'] + [f'{t}（节点{r + 1}）' for r in range(n) for t in ['预测值', '真实值', 'RMSE']]]
                df_train.to_excel(pjoin(sum_path, f'train_predictions_epoch_{epoch_i}.xlsx'), index=False)

        # # 保存验证集和测试集
        # df_val = pd.DataFrame(full_val_results)
        # df_val = df_val[
        #     ['组号', 'epoch'] + [f'{t}（节点{r + 1}）' for r in range(n) for t in ['预测值', '真实值', 'RMSE']]]
        # df_val.to_excel(pjoin(sum_path, f'val_predictions_epoch_{epoch_i}.xlsx'), index=False)
        #
        # df_test = pd.DataFrame(full_test_results)
        # df_test = df_test[
        #     ['组号', 'epoch'] + [f'{t}（节点{r + 1}）' for r in range(n) for t in ['预测值', '真实值', 'RMSE']]]
        # df_test.to_excel(pjoin(sum_path, f'test_predictions_epoch_{epoch_i}.xlsx'), index=False)
        #
        # df_train = pd.DataFrame(full_train_results)
        # df_train = df_train[
        #     ['组号', 'epoch'] + [f'{t}（节点{r + 1}）' for r in range(n) for t in ['预测值', '真实值', 'RMSE']]]
        # df_train.to_excel(pjoin(sum_path, f'train_predictions_epoch_{epoch_i}.xlsx'), index=False)

        writer.close()
    print('Training model finished!')

# 其他函数（model_inference、multi_pred等）保持原有实现不变