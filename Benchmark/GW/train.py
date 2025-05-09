# 在文件开头添加pandas导入
import torch
import numpy as np
import argparse
import time
import util

from engine import trainer
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/NewYork', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mat.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=11, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=100, help='')
# parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='./answer', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        outputs = []
        realy = torch.Tensor(dataloader['y_train']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            with torch.no_grad():
                preds = engine.model(trainx).transpose(1, 3)
            outputs.append(preds.squeeze())


            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)


        # validation
        # 在验证部分替换为以下代码（约第111行开始）
        # yhat = torch.cat(outputs, dim=0)
        # yhat = yhat[:realy.size(0), ...]
        # max_rows_per_file = 100000  # 每个Excel文件最大行数
        # file_counter = 1  # 文件计数器
        # current_rows = 0  # 当前文件行数计数器
        # all_data = []  # 当前文件数据暂存
        #
        # # 创建列头模板
        # columns = ['TimeStep', 'SampleID']
        # columns += [f'N{i + 1}_Pred' for i in range(11)] + [f'N{i + 1}_Real' for i in range(11)]
        #
        # for time_step in range(12):
        #     # 数据转换（确保在CPU上）
        #     pred = scaler.inverse_transform(yhat[:, :, time_step]).cpu().numpy()  # 形状 (num_samples, 11)
        #     real = realy[:, :, time_step].cpu().numpy()  # 形状 (num_samples, 11)
        #
        #     # 验证节点数量
        #     assert pred.shape[1] == 11, f"预期11个节点，实际得到{pred.shape[1]}个"
        #
        #     # 处理每个样本
        #     for sample_idx in range(pred.shape[0]):
        #         # 创建行数据
        #         row = {
        #             'TimeStep': time_step + 1,
        #             'SampleID': sample_idx + 1
        #         }
        #
        #         # 添加节点数据
        #         for node in range(11):
        #             row[f'N{node + 1}_Pred'] = pred[sample_idx, node]
        #             row[f'N{node + 1}_Real'] = real[sample_idx, node]
        #
        #         # 添加当前行到数据列表
        #         all_data.append(row)
        #         current_rows += 1
        #
        #         # 达到最大行数时保存文件
        #         if current_rows >= max_rows_per_file:
        #             # 创建DataFrame并保存
        #             df = pd.DataFrame(all_data)[columns]  # 保持列顺序
        #             df.to_excel(f"predictions_part_{file_counter}.xlsx", index=False)
        #
        #             # 重置计数器和缓存
        #             file_counter += 1
        #             current_rows = 0
        #             all_data = []
        #
        # # 保存剩余数据（最后未满10000行的部分）
        # if len(all_data) > 0:
        #     df = pd.DataFrame(all_data)[columns]
        #     df.to_excel(f"predictions_part1_{file_counter}.xlsx", index=False)
        #
        # print(f"共生成 {file_counter} 个Excel文件")



        # batch_count = 0  # 文件计数器
        # current_batch = []  # 当前批次数据
        #
        # for time_step in range(12):
        #     # 数据转换（确保在CPU上）
        #     pred = scaler.inverse_transform(yhat[:, :, time_step]).cpu().numpy()  # 形状 (num_samples, 11)
        #     real = realy[:, :, time_step].cpu().numpy()  # 形状 (num_samples, 11)
        #
        #     # 验证节点数量
        #     assert pred.shape[1] == 11, f"预期11个节点，实际得到{pred.shape[1]}个"
        #
        #     # 处理每个样本
        #     for sample_idx in range(pred.shape[0]):
        #         # 创建行数据
        #         row = {
        #             'TimeStep': time_step + 1,
        #             'SampleID': f"B{batch_count + 1}_S{sample_idx + 1}"  # 包含批次信息
        #         }
        #
        #         # 添加节点数据
        #         for node in range(11):
        #             row[f'N{node + 1}_Pred'] = pred[sample_idx, node]
        #             row[f'N{node + 1}_Real'] = real[sample_idx, node]
        #
        #         current_batch.append(row)
        #
        #         # 达到批次大小时保存文件
        #         if (sample_idx + 1) % args.batch_size == 0:
        #             # 创建DataFrame
        #             batch_df = pd.DataFrame(current_batch)
        #
        #             # 按时间步排序
        #             batch_df = batch_df.sort_values(by='TimeStep')
        #
        #             # 保存文件
        #             filename = f"prediction_batch_{batch_count + 1}.xlsx"
        #             batch_df.to_excel(filename, index=False)
        #
        #             # 重置批次
        #             current_batch = []
        #             batch_count += 1
        #
        # # 保存最后未满的批次
        # if len(current_batch) > 0:
        #     batch_df = pd.DataFrame(current_batch)
        #     filename = f"prediction_batch_{batch_count + 1}.xlsx"
        #     batch_df.to_excel(filename, index=False)

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    outputs = []
    realy = torch.Tensor(dataloader['y_train']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    # testing


    amae = []
    amape = []
    armse = []
    # 创建保存预测和真实值的容器
    all_preds = []
    all_reals = []

    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]

        # # 打印前3个样本的预测和真实值
        # print(f"\nHorizon {i+1} Predictions Sample:")
        # print("Prediction samples:", pred[:].cpu().numpy())
        # print("Ground truth samples:", real[:].cpu().numpy())
        #
        # 保存完整数据
        all_preds.append(pred.cpu().numpy())
        all_reals.append(real.cpu().numpy())

        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

