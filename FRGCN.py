import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np

# 参数设置
# 参数设置部分修改
M = 3  # 改为3个节点
Nt = 93
dx = 1.0
dt = 0.05
target_city = 5  # 目标城市0-10
target_col = 6  # 数据列1-11

# 读取文件数据
node = pd.read_csv("NYC_node_attr.csv")
cities = node.index[1:].astype(int).tolist()  # 城市标签为 0-10

# 定位城市
city_row = node.loc[target_city][1:]

related = city_row[city_row != 0]  # .sort_values(ascending=False)
top3_cities = related.head(2).index.astype(int).tolist()  # 获取前2个关联城市
print(f"目标城市 {target_city} 的关联城市:", top3_cities)
related_cols = [c + 1 for c in top3_cities]

# 只读取需要的列（ 目标城市 + 关联城市）
cols_to_load = [target_col] + related_cols
data_all = pd.read_excel("NYC_data_11_nodes.xlsx")
column_names = [data_all.columns[i] for i in cols_to_load]
data = data_all[column_names]
data.columns = ["L_j0"] + [f"L_j{i + 1}" for i in range(2)]
print("读取完毕，开始分组训练......")


def create_laplacian(M, dx):
    # 仅生成边索引，不计算归一化权重
    edge_index = torch.tensor([[i, j] for i in range(M) for j in range(M)], dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=M)
    return edge_index


def gl_weights(alpha, edge_index):
    if isinstance(alpha, torch.Tensor) and alpha.dim() == 0:
        alpha = alpha.unsqueeze(0)  # 将标量转为向量
    row, col = edge_index
    edge_weights = alpha[row] * alpha[col]  # 确保alpha是向量
    return edge_weights


class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat_conv = GATConv(hidden_dim, hidden_dim, heads=1, concat=True)
        self.dropout = nn.Dropout(0.2)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.gat_conv(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x

gcn_model = GCNModel(input_dim=3, hidden_dim=64, output_dim=3)

def solve_fpde(alphas, Nt, dt, L, u0, f):
    batch_size = alphas.shape[0]
    # 调整初始维度为四维 [batch_size, 1, M, 3]
    u_hist = [u0.unsqueeze(1).unsqueeze(-1).expand(-1, -1, M, 3)]
    # 计算度矩阵归一化系数
    for n in range(Nt):
        # 生成动态边权重（结合alpha和归一化）
        edge_weight_alpha = gl_weights(alphas, L)
        #print(edge_weight_alpha)
        edge_weight = norm * edge_weight_alpha  # 结合归一化与alpha权重
        if n == 0:
            # 保持维度一致性 [batch_size*M, 3]
            terms = []
            for b in range(3):
                terms.append(u_hist[0][b] * alphas[b])
            if terms:
                stacked_terms = torch.stack(terms)
                hist_sum = stacked_terms.view(-1, M, 3).sum(dim=0)
                hist_sum = hist_sum.repeat(batch_size, 1)  # [batch_size*M, 3]
            # hist_sum = torch.zeros((batch_size * M, 3), dtype=u_hist[0].dtype)
        else:
            terms = []
            for b in range(batch_size):
                for k in range(1, min(n + 1, Nt)):
                    if n - k >= 0 and n - k < len(u_hist):
                        history = u_hist[n - k].squeeze(1)[b]  # 保持维度 [M, 3]
                        terms.append(edge_weight_alpha[-1] * history)
            if terms:
                stacked_terms = torch.stack(terms)
                hist_sum = stacked_terms.view(-1, M, 3).sum(dim=0)
                hist_sum = hist_sum.repeat(batch_size, 1)  # [batch_size*M, 3]
            else:
                hist_sum = torch.zeros((batch_size * M, 3), dtype=u_hist[0].dtype)
        # 使用GCN进行预测
        u_n = gcn_model(hist_sum.view(batch_size * M, 3), L, edge_weight)
        u_hist.append(u_n.view(batch_size, 1, M, 3))
    return torch.cat(u_hist[1:Nt + 1], dim=1)


# 分组循环
group_size = 96
results = []
good_result = []
L = create_laplacian(M, dx)
row, col = L
deg = degree(row, M, dtype=torch.float32)
deg_inv_sqrt = deg.pow(-0.5)
norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

save_interval = 100000  # 新增：每10万行保存一次
total_processed = 0    # 新增：累计处理行数计数器
file_counter = 1        # 新增：文件序号
try:
    for group_num, i in enumerate(range(0,10000, group_size)):  # len(data)
        print(f"Processing group {group_num}")
        group = data.iloc[i:i + group_size]

        # 检查分组完整性
        current_group_size = len(group)
        if (i + group_size > len(data)) and (current_group_size < group_size):
            raise ValueError(f"最后一组元素不足，预期 {group_size} 个，实际 {current_group_size} 个")
        # 更新累计处理行数
        total_processed += current_group_size  # 新增
        # 检查初始数据是否足够
        if len(group) < 3:
            raise ValueError("初始时间步数据不足，至少需要3个时间步")

        L_j0 = group["L_j0"].values
        L_j1 = group["L_j1"].values
        L_j2 = group["L_j2"].values
        # 初始化三个节点的初始条件（使用前3个时间步）
        u0 = torch.stack([
            torch.tensor(L_j0[:3], dtype=torch.float32),
            torch.tensor(L_j1[:3], dtype=torch.float32),
            torch.tensor(L_j2[:3], dtype=torch.float32)
        ])  # 形状 [3节点, 3时间步]
        # 观测数据使用后续时间步（3节点 × 31时间步）
        u_obs_raw = torch.stack([
            torch.tensor(L_j0[3:Nt + 3], dtype=torch.float32),
            torch.tensor(L_j1[3:Nt + 3], dtype=torch.float32),
            torch.tensor(L_j2[3:Nt + 3], dtype=torch.float32)
        ])
        # 归一化
        u_obs = (u_obs_raw - u_obs_raw.mean(dim=1, keepdim=True)) / \
                (u_obs_raw.std(dim=1, keepdim=True) + 1e-8)
        # 修改维度扩展方式
        u_obs1 = u_obs_raw.view(3, Nt, 1).expand(-1, -1, 3)  # 形状 [3, 31, 3, 3]
        # u_obs1 = u_obs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M, 3)  # 形状 [3, 31, 3, 3]
        u_obs = u_obs.unsqueeze(-1).expand(-1, -1, 3)

        # 修改参数初始化，将 alpha 初始化为三个独立的 Parameter
        alpha = nn.Parameter(torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32))
        b = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        optimizer = optim.Adam([
            {'params': alpha, 'lr': 0.01},
            {'params': gcn_model.parameters(), 'lr': 0.001}, {'params': b, 'lr': 0.005}], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # 早停
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        # 修改solve_fpde调用方式（在训练循环中）
        for epoch in range(100):
            optimizer.zero_grad()
            alpha_tensor = torch.stack([alpha[0], alpha[1], alpha[2]])
            # print(f"Epoch {epoch}: Alpha={alpha.detach().numpy()}")
            x = torch.arange(M, dtype=torch.float32)
            f_values = x * torch.sin(x) + b
            f = f_values.expand(Nt, -1)
            u_pred = solve_fpde(alpha_tensor, Nt, dt, L, u0, f)
            u_pred1 = u_pred[:, :, :, 0].squeeze(-1)
            u_pred_loss = u_pred1 * (u_obs1.max(dim=1, keepdim=True)[0] - u_obs1.min(dim=1, keepdim=True)[0]) \
                          + u_obs1.min(dim=1, keepdim=True)[0]
            # 修正损失函数
            loss = torch.mean(torch.abs(u_pred_loss[:, :, [0]] - u_obs1[:, :, [0]]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=1.0)
            scheduler.step(loss)
            optimizer.step()

            # 保持alpha在合理范围
            with torch.no_grad():
                for a in alpha:
                    a.clamp_(0.005, 0.995)
                # 早停检查
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    no_improve = 0
                else:
                    no_improve += 1
                # 触发早停
                if no_improve >= patience:
                    print(f'\nEarly stopping triggered at epoch {epoch} with loss {current_loss:.4f}')
                    break  # 退出训练循环

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, '
                      f'alphas: {alpha[0].item():.4f}, {alpha[1].item():.4f}, {alpha[2].item():.4f}')

        print(f'Optimized alphas: '
              f'L_j0: {alpha[0].item():.4f}, '
              f'L_j1: {alpha[1].item():.4f}, '
              f'L_j2: {alpha[2].item():.4f}')

        with torch.no_grad():
            L = create_laplacian(M, dx)
            x = torch.arange(M, dtype=torch.float32) * dx
            f_values = x * torch.sin(x) + b.detach()
            f = f_values.expand(Nt, -1)

            u_pred_final = solve_fpde(alpha, Nt, dt, L, u0, f)

            # 确保 u_obs 的维度与 u_pred_final 一致
            u_obs = u_obs.view(3, Nt, 3)  # 调整 u_obs 的维度为 [3, 31, 3]
            u_obs_raw = u_obs_raw.view(3, Nt, 1).expand(-1, -1, 3)

            # 反归一化时使用每个时间步自身的最大最小值
            max_vals = u_obs_raw.max(dim=1, keepdim=True)[0]  # 维度 [3,1,3]
            min_vals = u_obs_raw.min(dim=1, keepdim=True)[0]
            u_pred_denorm = u_pred_final[:, :, 0, :] * (max_vals - min_vals) + min_vals

            u_pred_denorm = u_pred_denorm.view(3, Nt, 3)  # 将四维 [3, 93, 3, 3] 转为三维 [3, 93, 3]

            final_mse = (torch.mean(torch.abs(u_pred_loss[:, :, [0]] - u_obs1[:, :, [0]]))).item()
            # final_mse = (torch.mean((u_pred_denorm - u_obs_raw) ** 2) ).item()
        print(f'Final MSE: {final_mse:.4f}')
        results.append([group_num + 1, final_mse,np.sqrt(final_mse), alpha[0].item(), alpha[1].item(), alpha[2].item()])
        for m in range(Nt):
            if final_mse < 10000:
                rmse0 = np.sqrt((u_pred_denorm[0, m, 0] - u_obs_raw[0, m, 0]) ** 2)
                rmse1 = np.sqrt((u_pred_denorm[1, m, 0] - u_obs_raw[1, m, 0]) ** 2)
                rmse2 = np.sqrt((u_pred_denorm[2, m, 0] - u_obs_raw[2, m, 0]) ** 2)
                good_result.append([group_num + 1, L_j0[m + 3], f"{u_pred_denorm[0, m, 0].detach().numpy():.2f}",
                                    f"{rmse0.detach().numpy():.2f}",
                                    L_j1[m + 3], f"{u_pred_denorm[1, m, 0].detach().numpy():.2f}",
                                    f"{rmse1.detach().numpy():.2f}",
                                    L_j2[m + 3], f"{u_pred_denorm[2, m, 0].detach().numpy():.2f}",
                                    f"{rmse2.detach().numpy():.2f}"])

        if total_processed >= save_interval * file_counter:
            # 保存当前结果
            df = pd.DataFrame(results, columns=['组号', 'MSE', 'RMSE', 'Alpha0', 'Alpha1', 'Alpha2'])
            df_data = pd.DataFrame(good_result,
                                   columns=['组号', '真实值L_j0', '预测值L_j0', 'rmse0', '真实值L_j1', '预测值L_j1',
                                            'rmse1','真实值L_j2', '预测值L_j2', 'rmse2'])

            # 生成带序号的文件名
            df.to_excel(f'results_part6_{file_counter}.xlsx', index=False)
            df_data.to_excel(f'results_data_part6_{file_counter}.xlsx', index=False)
            print(f"已保存第 {file_counter} 批结果，处理行数：{total_processed}")

            # 重置存储和计数器
            results.clear()
            good_result.clear()
            file_counter += 1

except ValueError as e:
    print(f"\n错误发生: {str(e)}")
finally:
    # 最终保存剩余数据
    if len(results) > 0:
        df = pd.DataFrame(results, columns=['组号', 'MSE', 'RMSE', 'Alpha0', 'Alpha1', 'Alpha2'])
        df_data = pd.DataFrame(good_result,columns=['组号', '真实值L_j0', '预测值L_j0', 'rmse0', '真实值L_j1', '预测值L_j1',
                                        'rmse1','真实值L_j2', '预测值L_j2', 'rmse2'])
        df.to_excel(f'results_part6_{file_counter}.xlsx', index=False)
        df_data.to_excel(f'results_data_part6_{file_counter}.xlsx', index=False)
        print(f"已保存最后一批结果，处理行数：{total_processed}")
    print("所有结果已保存至Excel文件")


