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
    col_code = 'code'
    col_day = 'day'
    col_target = 'return'
    col_node_idx = 'node_idx'
    feature_start_idx = 3  # 特征起始列索引

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
    """设置随机种子确保可复现性"""
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
    """重定向标准输出到日志文件"""

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


def setup_experiment(config):
    """初始化实验目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{config.project_name}"
    run_dir = os.path.join(config.base_output_dir, dir_name)

    # 创建标准目录结构
    dirs = {
        "checkpoints": "checkpoints",
        "logs": "logs",
        "results": "results"
    }

    paths = {}
    for key, name in dirs.items():
        path = os.path.join(run_dir, name)
        os.makedirs(path, exist_ok=True)
        paths[key] = path

    # 重定向输出
    sys.stdout = Logger(os.path.join(paths["logs"], 'train_log.txt'), sys.stdout)

    print(f"=== 实验启动: {timestamp} ===")
    print(f"输出目录: {run_dir}")
    print(f"设备: {config.device}")
    print(f"数据路径: {config.data_path}")

    return run_dir, paths["checkpoints"], paths["logs"], paths["results"]


def plot_history(history, save_dir):
    """绘制训练历史曲线（只保存一次）"""
    plt.style.use('bmh')
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(16, 6))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', alpha=0.8)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # IC 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_ic'], 'r--', label='Train IC', alpha=0.5)
    plt.plot(epochs, history['valid_ic'], 'g-s', label='Valid IC', linewidth=2)

    # 标注最佳点
    if history['valid_ic']:
        max_ic = max(history['valid_ic'])
        max_epoch = history['valid_ic'].index(max_ic) + 1
        plt.annotate(f'Best: {max_ic:.4f}', xy=(max_epoch, max_ic),
                     xytext=(max_epoch, max_ic + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title('IC Curve (Information Coefficient)')
    plt.xlabel('Epoch')
    plt.ylabel('IC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存到 logs 目录
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"训练曲线图已保存: {plot_path}")
    plt.close()


# ==========================================
# 3. 数据处理 (Data Pipeline)
# ==========================================
class TimeSeriesDataset:
    """时序数据集处理类"""

    def __init__(self, config):
        print(f"加载数据: {config.data_path} ...")
        if not os.path.exists(config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {config.data_path}")

        df = pd.read_parquet(config.data_path)
        self._validate_data(df, config)

        # 自动处理特征列（排除node_idx）
        self.feature_cols = [col for col in df.columns
                             if col not in [config.col_node_idx, config.col_code, config.col_day, config.col_target]]
        self.num_features = len(self.feature_cols)
        self.num_nodes = df[config.col_node_idx].max() + 1

        # 时间索引映射
        self.day_map = {day: idx for idx, day in enumerate(sorted(df[config.col_day].unique()))}
        self.total_days = len(self.day_map)

        print(f"数据集统计: 节点数={self.num_nodes}, 总天数={self.total_days}, 特征维度={self.num_features}")
        print("构建全局内存张量 (Pinned Memory)...")
        self._build_global_tensors(df, config)

    def _validate_data(self, df, config):
        """验证数据格式"""
        required_cols = [config.col_code, config.col_day, config.col_target, config.col_node_idx]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {', '.join(missing)}")

    def _build_global_tensors(self, df, config):
        """构建全局特征和目标张量"""
        self.global_features = torch.zeros(
            (self.num_nodes, self.total_days, self.num_features),
            dtype=torch.float32
        )
        self.global_targets = torch.zeros(
            (self.num_nodes, self.total_days),
            dtype=torch.float32
        )
        self.active_mask = torch.zeros(
            (self.num_nodes, self.total_days),
            dtype=torch.bool
        )

        # 按日期分组填充
        for day, group in tqdm(df.groupby(config.col_day), desc="填充张量"):
            day_idx = self.day_map[day]
            node_idxs = group[config.col_node_idx].values

            # 提取特征和目标
            feats = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)
            rets = torch.tensor(group[config.col_target].values, dtype=torch.float32)

            # 填充张量
            self.global_features[node_idxs, day_idx, :] = feats
            self.global_targets[node_idxs, day_idx] = rets
            self.active_mask[node_idxs, day_idx] = True

    def get_batch(self, day_idx, window):
        """获取指定时间点的批次数据"""
        if day_idx < window:
            return None

        # 获取当天活跃节点
        mask = self.active_mask[:, day_idx]
        active_indices = torch.where(mask)[0]
        if len(active_indices) == 0:
            return None

        # 提取窗口数据
        start = day_idx - window + 1
        end = day_idx + 1
        x = self.global_features[active_indices, start:end, :]
        y = self.global_targets[active_indices, day_idx]

        return x, y, active_indices


# ==========================================
# 4. 模型定义 (Model: GRU + AdaGNN)
# ==========================================
class AdaptiveGraphLearner(nn.Module):
    """自适应图学习器"""

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
        row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        return adj / row_sum


class GNNLayer(nn.Module):
    """图卷积层"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x, adj):
        h = self.linear(x)
        h_prime = torch.mm(adj, h)
        return self.act(h_prime)


class TemporalAdaGNN(nn.Module):
    """时序自适应图神经网络"""

    def __init__(self, num_nodes, input_dim, config):
        super().__init__()
        # 时序模块
        self.gru = nn.GRU(input_dim, config.hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(config.hidden_dim)

        # 图模块
        self.graph_learner = AdaptiveGraphLearner(num_nodes, config.node_emb_dim)
        self.gnn = GNNLayer(config.hidden_dim, config.hidden_dim)

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, active_indices):
        """前向传播"""
        self.gru.flatten_parameters()
        out, hn = self.gru(x)
        h_self = hn[-1]  # 最后时刻隐藏状态
        h_self = self.bn(h_self)

        adj = self.graph_learner(active_indices)
        h_graph = self.gnn(h_self, adj)

        h_cat = torch.cat([h_self, h_graph], dim=1)
        return self.head(h_cat).squeeze()


# ==========================================
# 5. 训练主程序
# ==========================================
def ic_loss(pred, target):
    """信息系数损失函数"""
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return 1 - corr


def calc_ic(pred, target):
    """计算信息系数"""
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return corr.item()


def train():
    """训练主函数"""
    # 1. 设置实验目录
    run_dir, ckpt_dir, log_dir, result_dir = setup_experiment(Config)

    # 2. 准备数据
    dataset = TimeSeriesDataset(Config)

    # 验证数据划分
    train_end = Config.train_days
    valid_end = train_end + Config.valid_days

    if valid_end > dataset.total_days:
        print(f"警告: 验证集结束日期 {valid_end} 超出数据范围 {dataset.total_days}，已调整")
        valid_end = dataset.total_days

    if train_end > dataset.total_days:
        raise ValueError(f"训练集天数 {Config.train_days} 超出总天数 {dataset.total_days}")

    # 3. 初始化模型
    model = TemporalAdaGNN(dataset.num_nodes, dataset.num_features, Config).to(Config.device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # 4. 训练历史
    history = {
        'train_loss': [],
        'train_ic': [],
        'valid_ic': []
    }
    best_ic = -999.0

    print("\n>>> 开始训练...")

    for epoch in range(Config.num_epochs):
        # 训练模式
        model.train()
        epoch_loss = []
        epoch_ic = []

        # 训练循环
        pbar = tqdm(
            range(Config.window_size, train_end),
            desc=f"Epoch {epoch + 1}/{Config.num_epochs}",
            leave=False
        )
        for day_idx in pbar:
            batch = dataset.get_batch(day_idx, Config.window_size)
            if batch is None:
                continue

            x, y, idx = batch
            x, y, idx = x.to(Config.device), y.to(Config.device), idx.to(Config.device)

            optimizer.zero_grad()
            pred = model(x, idx)

            # 组合损失
            l_mse = F.mse_loss(pred, y)
            l_ic = ic_loss(pred, y)
            loss = (1 - Config.ic_loss_weight) * l_mse + Config.ic_loss_weight * l_ic

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_ic.append(calc_ic(pred, y))
            pbar.set_postfix({'IC': f'{epoch_ic[-1]:.4f}'})

        # 验证模式
        model.eval()
        val_ics = []
        with torch.no_grad():
            for day_idx in range(train_end, valid_end):
                batch = dataset.get_batch(day_idx, Config.window_size)
                if batch is None:
                    continue
                x, y, idx = batch
                x, y, idx = x.to(Config.device), y.to(Config.device), idx.to(Config.device)
                pred = model(x, idx)
                val_ics.append(calc_ic(pred, y))

        # 记录指标
        avg_loss = np.mean(epoch_loss)
        avg_trn_ic = np.mean(epoch_ic) if epoch_ic else 0.0
        avg_val_ic = np.mean(val_ics) if val_ics else 0.0

        history['train_loss'].append(avg_loss)
        history['train_ic'].append(avg_trn_ic)
        history['valid_ic'].append(avg_val_ic)

        print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | Train IC: {avg_trn_ic:.4f} | Valid IC: {avg_val_ic:.4f}")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:02d}.pth"))

        # 保存最佳模型
        if avg_val_ic > best_ic:
            best_ic = avg_val_ic
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print(f"  >>> 新最佳模型保存: {run_dir}/best_model.pth")

    # 5. 训练结束
    print(f"\n训练完成! 最佳验证 IC: {best_ic:.4f}")
    plot_history(history, log_dir)  # 只保存一次曲线图
    print("所有结果已保存完毕。")
    print(f"实验目录: {run_dir}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback

        traceback.print_exc()