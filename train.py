import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import shutil


# ==========================================
# 1. 基础配置 (Configuration)
# ==========================================
class Config:
    # --- 基础环境 ---
    project_name = "Temporal_AdaGNN"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据路径 ---
    data_path = "366FactorsData_standard.parquet"

    # --- [关键] 字段定义 ---
    # 假设前3列是 metadata，特征从第3列(索引3)开始
    # 且 node_idx 在最后一列，代码会自动处理
    col_code = 'code'
    col_day = 'day'
    col_target = 'return'
    col_node_idx = 'node_idx'
    feature_start_idx = 3

    # --- 时序设置 ---
    window_size = 20  # 回看窗口 (T-19 ... T)

    # --- 数据划分 ---
    train_days = 600
    valid_days = 100

    # --- 模型超参数 ---
    hidden_dim = 64
    node_emb_dim = 32
    dropout = 0.5

    # --- 训练参数 ---
    num_epochs = 15
    learning_rate = 1e-3
    weight_decay = 1e-4
    ic_loss_weight = 0.8  # 强 IC 导向

    # --- 输出根目录 ---
    base_output_dir = "runs"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


set_seed(Config.seed)


# ==========================================
# 2. 日志与路径管理 (Logger & Utils)
# ==========================================
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# 在 setup_experiment 函数中添加新的目录结构
def setup_experiment(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{config.project_name}"
    run_dir = os.path.join(config.base_output_dir, dir_name)

    # 新增的目录结构
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")  # 日志和图片目录
    result_dir = os.path.join(run_dir, "results")  # CSV结果目录

    for directory in [ckpt_dir, log_dir, result_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 重定向输出到日志文件夹
    sys.stdout = Logger(os.path.join(log_dir, 'train_log.txt'), sys.stdout)

    print(f"=== 实验启动: {timestamp} ===")
    print(f"输出目录: {run_dir}")
    print(f"设备: {config.device}")

    return run_dir, ckpt_dir, log_dir, result_dir



def plot_history(history, save_dir):
    plt.style.use('bmh')  # 使用一种美观的样式
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(16, 6))

    # 子图1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', alpha=0.8)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2: IC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_ic'], 'r--', label='Train IC', alpha=0.5)
    plt.plot(epochs, history['valid_ic'], 'g-s', label='Valid IC', linewidth=2)

    # 标注最佳点
    if len(history['valid_ic']) > 0:
        max_ic = max(history['valid_ic'])
        max_epoch = history['valid_ic'].index(max_ic) + 1
        plt.annotate(f'Best: {max_ic:.4f}', xy=(max_epoch, max_ic), xytext=(max_epoch, max_ic + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title('IC Curve (Information Coefficient)')
    plt.xlabel('Epoch')
    plt.ylabel('IC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')  # 保存到 logs 目录
    plt.savefig(plot_path, dpi=300)
    print(f"曲线图已保存: {plot_path}")
    plt.close()


# ==========================================
# 3. 数据处理 (Data Pipeline)
# ==========================================
class TimeSeriesDataset:
    def __init__(self, config):
        print(f"正在加载数据: {config.data_path} ...")
        if not os.path.exists(config.data_path):
            raise FileNotFoundError(f"找不到文件: {config.data_path}")

        df = pd.read_parquet(config.data_path)

        # --- 自动处理特征列 (排除最后一列 node_idx) ---
        all_cols = df.columns
        # 1. 获取从 feature_start_idx 开始的所有列
        potential_feats = all_cols[config.feature_start_idx:].tolist()

        # 2. 必须把 node_idx 排除掉 (因为它在最后一列)
        if config.col_node_idx in potential_feats:
            potential_feats.remove(config.col_node_idx)

        self.feature_cols = potential_feats
        self.num_features = len(self.feature_cols)

        # 获取节点数
        self.num_nodes = df[config.col_node_idx].max() + 1

        # 时间映射
        unique_days = sorted(df[config.col_day].unique())
        self.day_map = {day: i for i, day in enumerate(unique_days)}
        self.total_days = len(unique_days)

        print(f"数据集统计: 节点数={self.num_nodes}, 总天数={self.total_days}, 特征维度={self.num_features}")
        print("正在构建全局内存张量 (Pinned Memory)...")

        # 初始化大张量
        self.global_features = torch.zeros((self.num_nodes, self.total_days, self.num_features), dtype=torch.float32)
        self.global_targets = torch.zeros((self.num_nodes, self.total_days), dtype=torch.float32)
        self.active_mask = torch.zeros((self.num_nodes, self.total_days), dtype=torch.bool)

        # 填充数据
        grouped = df.groupby(config.col_day)
        for day, group in tqdm(grouped, desc="Filling Tensor"):
            d_idx = self.day_map[day]

            node_idxs = group[config.col_node_idx].values

            # 严格提取特征列
            feats = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)
            rets = torch.tensor(group[config.col_target].values, dtype=torch.float32)

            self.global_features[node_idxs, d_idx, :] = feats
            self.global_targets[node_idxs, d_idx] = rets
            self.active_mask[node_idxs, d_idx] = True

        print("数据构建完毕。")

    def get_batch(self, day_idx, window):
        """
        获取 Batch 数据
        """
        if day_idx < window: return None

        # 1. 只有当天存在的股票才参与训练/预测
        mask = self.active_mask[:, day_idx]
        active_indices = torch.where(mask)[0]

        if len(active_indices) == 0: return None

        # 2. 切片: T-Window+1 到 T (假设因子已对齐到 T 时刻)
        # 输入: 过去 window 天的因子序列
        start = day_idx - window + 1
        end = day_idx + 1

        x = self.global_features[active_indices, start:end, :]
        y = self.global_targets[active_indices, day_idx]

        return x, y, active_indices


# ==========================================
# 4. 模型定义 (Model: GRU + AdaGNN)
# ==========================================
class AdaptiveGraphLearner(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.relu = nn.ReLU()

    def forward(self, active_indices):
        emb1 = self.node_emb1[active_indices]
        emb2 = self.node_emb2[active_indices]
        adj = torch.mm(emb1, emb2.t())
        adj = self.relu(adj)
        # Row Normalize
        row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        adj = adj / row_sum
        return adj


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x, adj):
        h = self.linear(x)
        h_prime = torch.mm(adj, h)
        return self.act(h_prime)


class TemporalAdaGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, config):
        super().__init__()
        # 时序模块
        self.gru = nn.GRU(input_dim, config.hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(config.hidden_dim)

        # 图模块
        self.graph_learner = AdaptiveGraphLearner(num_nodes, config.node_emb_dim)
        self.gnn = GNNLayer(config.hidden_dim, config.hidden_dim)

        # 预测模块
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, active_indices):
        # x: (Batch, Window, Feat)
        self.gru.flatten_parameters()
        out, hn = self.gru(x)
        h_self = hn[-1]  # 取最后一步 hidden state
        h_self = self.bn(h_self)

        adj = self.graph_learner(active_indices)
        h_graph = self.gnn(h_self, adj)

        h_cat = torch.cat([h_self, h_graph], dim=1)
        return self.head(h_cat).squeeze()


# ==========================================
# 5. 训练主程序
# ==========================================
def ic_loss(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return 1 - corr


def calc_ic(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return corr.item()


def train():
    # 1. 设置实验目录
    run_dir, ckpt_dir, log_dir, result_dir = setup_experiment(Config)

    # 2. 准备数据
    dataset = TimeSeriesDataset(Config)

    # 索引计算
    start_idx = Config.window_size
    train_end = Config.train_days
    valid_end = train_end + Config.valid_days

    if valid_end > dataset.total_days:
        print(f"Warning: Valid end {valid_end} exceeds data length {dataset.total_days}. Adjusting...")
        valid_end = dataset.total_days

    # 3. 初始化模型
    model = TemporalAdaGNN(dataset.num_nodes, dataset.num_features, Config).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    history = {'train_loss': [], 'train_ic': [], 'valid_ic': []}
    best_ic = -999.0

    print("\n>>> 开始训练...")

    for epoch in range(Config.num_epochs):
        model.train()
        epoch_loss = []
        epoch_ic = []

        # --- Train Loop ---
        pbar = tqdm(range(start_idx, train_end), desc=f"Epoch {epoch + 1}/{Config.num_epochs}", leave=False)
        for day_idx in pbar:
            batch = dataset.get_batch(day_idx, Config.window_size)
            if batch is None: continue

            x, y, idx = batch
            x, y, idx = x.to(Config.device), y.to(Config.device), idx.to(Config.device)

            optimizer.zero_grad()
            pred = model(x, idx)

            l_mse = F.mse_loss(pred, y)
            l_ic = ic_loss(pred, y)
            loss = (1 - Config.ic_loss_weight) * l_mse + Config.ic_loss_weight * l_ic

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_ic.append(1 - l_ic.item())
            pbar.set_postfix({'IC': f'{epoch_ic[-1]:.4f}'})

        # --- Valid Loop ---
        model.eval()
        val_ics = []
        with torch.no_grad():
            for day_idx in range(train_end, valid_end):
                batch = dataset.get_batch(day_idx, Config.window_size)
                if batch is None: continue
                x, y, idx = batch
                x, y, idx = x.to(Config.device), y.to(Config.device), idx.to(Config.device)
                pred = model(x, idx)
                val_ics.append(calc_ic(pred, y))

        # --- 统计与保存 ---
        avg_loss = np.mean(epoch_loss)
        avg_trn_ic = np.mean(epoch_ic)
        avg_val_ic = np.mean(val_ics) if val_ics else 0.0

        history['train_loss'].append(avg_loss)
        history['train_ic'].append(avg_trn_ic)
        history['valid_ic'].append(avg_val_ic)

        print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | Train IC: {avg_trn_ic:.4f} | Valid IC: {avg_val_ic:.4f}")

        # [关键] 每一轮都保存模型
        epoch_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:02d}.pth")
        torch.save(model.state_dict(), epoch_path)

        # 修改保存最佳模型的位置（仍保留在 run_dir 根目录）
        if avg_val_ic > best_ic:
            best_ic = avg_val_ic
            best_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  >>> New Best Found! Saved to {best_path}")
        plot_history(history, log_dir)  # 保存到 logs 目录

    # 4. 结束工作
    print(f"\n训练结束. Best Valid IC: {best_ic:.4f}")
    plot_history(history, run_dir)
    print("所有结果已保存完毕。")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()