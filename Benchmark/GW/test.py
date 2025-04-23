import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/NewYork',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mat.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=11,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')


args = parser.parse_args()




def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    # realy = torch.Tensor(dataloader['y_train']).to(device)
    # realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = torch.Tensor(dataloader['y_test']).to(device)

    realy = realy.transpose(1,3)[:,0,:,:]
    # for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
    # for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())


    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    max_rows_per_file = 100000  # 每个Excel文件最大行数
    file_counter = 1  # 文件计数器
    current_rows = 0  # 当前文件行数计数器
    all_data = []  # 当前文件数据暂存

    # 创建列头模板
    columns = ['TimeStep', 'SampleID']
    columns += [f'N{i + 1}_Pred' for i in range(11)] + [f'N{i + 1}_Real' for i in range(11)]

    for time_step in range(12):
        # 数据转换（确保在CPU上）
        pred = scaler.inverse_transform(yhat[:, :, time_step]).cpu().numpy()  # 形状 (num_samples, 11)
        real = realy[:, :, time_step].cpu().numpy()  # 形状 (num_samples, 11)

        # 验证节点数量
        assert pred.shape[1] == 11, f"预期11个节点，实际得到{pred.shape[1]}个"

        # 处理每个样本
        for sample_idx in range(pred.shape[0]):
            # 创建行数据
            row = {
                'TimeStep': time_step + 1,
                'SampleID': sample_idx + 1
            }

            # 添加节点数据
            for node in range(11):
                row[f'N{node + 1}_Pred'] = pred[sample_idx, node]
                row[f'N{node + 1}_Real'] = real[sample_idx, node]

            # 添加当前行到数据列表
            all_data.append(row)
            current_rows += 1

            # 达到最大行数时保存文件
            if current_rows >= max_rows_per_file:
                # 创建DataFrame并保存
                df = pd.DataFrame(all_data)[columns]  # 保持列顺序
                # df.to_excel(f"train_part_{file_counter}.xlsx", index=False)
                # df.to_excel(f"val_part_{file_counter}.xlsx", index=False)
                df.to_excel(f"test_part_{file_counter}.xlsx", index=False)
                # 重置计数器和缓存
                file_counter += 1
                current_rows = 0
                all_data = []

    # 保存剩余数据（最后未满10000行的部分）
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)[columns]
        df.to_excel(f"predictions_partl_{file_counter}.xlsx", index=False)


    print(f"共生成 {file_counter} 个Excel文件")

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))

    # y12 = realy[:, :, 1].squeeze().cpu().detach().numpy()
    # yhat12 = scaler.inverse_transform(yhat[:, :, 1].squeeze()).cpu().detach().numpy()
    #
    # y3 = realy[:, :, 2].squeeze().cpu().detach().numpy()
    # yhat3 = scaler.inverse_transform(yhat[:, :, 2].squeeze()).cpu().detach().numpy()
    # print("y12 shape:", y12.shape)
    # print("yhat12 shape:", yhat12.shape)
    # print("y3 shape:", y3.shape)
    # print("yhat3 shape:", yhat3.shape)
    # # 假设需要将二维数组的所有列拼接成一维（例如按行拼接）
    # y12_flat = y12.reshape(-1)  # 展平为 (324030*11,)
    # yhat12_flat = yhat12.reshape(-1)
    # y3_flat = y3.reshape(-1)
    # yhat3_flat = yhat3.reshape(-1)

    # 创建DataFrame
    # df2 = pd.DataFrame({
    #     'real12': y12_flat,
    #     'pred12': yhat12_flat,
    #     'real3': y3_flat,
    #     'pred3': yhat3_flat
    # })
    # # df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    # df2.to_excel('./wave.xlsx',index=False)


if __name__ == "__main__":
    main()
