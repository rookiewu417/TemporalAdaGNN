import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import math
import gc  # 垃圾回收模块


# ==========================================
# 0. 全局设置
# ==========================================
def set_plot_style():
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=fonts[0] if sys.platform == 'win32' else fonts[2])
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})


set_plot_style()


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

    def close(self):
        self.log.close()


# ==========================================
# 1. 配置 (Config)
# ==========================================
class Config:
    # model_type 将在运行时动态修改
    model_type = "Temp"

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "366FactorsData_standard.parquet"
    base_output_dir = "runs_baselines"

    col_code = 'code';
    col_day = 'day';
    col_target = 'return';
    col_node_idx = 'node_idx'
    feature_start_idx = 3
    window_size = 20
    train_days = 600
    valid_days = 100

    hidden_dim = 64
    dropout = 0.5

    # Transformer 专用参数
    n_head = 4
    n_layers = 2

    num_epochs = 15
    learning_rate = 1e-3
    weight_decay = 1e-4
    ic_loss_weight = 0.8


# ==========================================
# 2. 模型定义 (四种 Baseline)
# ==========================================

# --- 1. MLP ---
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, x):
        # x: (Batch, Window, Feat) -> 取最后一个时间步
        return self.net(x[:, -1, :]).squeeze()


# --- 2. LSTM ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, config.hidden_dim, num_layers=2, batch_first=True, dropout=config.dropout)
        self.bn = nn.BatchNorm1d(config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, (hn, cn) = self.lstm(x)
        return self.head(self.bn(hn[-1])).squeeze()


# --- 3. GRU ---
class BaselineGRU(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.gru = nn.GRU(input_dim, config.hidden_dim, num_layers=2, batch_first=True, dropout=config.dropout)
        self.bn = nn.BatchNorm1d(config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        self.gru.flatten_parameters()
        out, hn = self.gru(x)
        return self.head(self.bn(hn[-1])).squeeze()


# --- 4. Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BaselineTransformer(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim, max_len=config.window_size + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.n_head,
            dim_feedforward=config.hidden_dim * 2, dropout=config.dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.bn = nn.BatchNorm1d(config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last_step = out[:, -1, :]
        return self.head(self.bn(last_step)).squeeze()


# ==========================================
# 3. 数据处理
# ==========================================
class TimeSeriesDataset:
    def __init__(self, config):
        print(f"Loading Data: {config.data_path}")
        df = pd.read_parquet(config.data_path)

        # 强制日期转换逻辑
        if pd.api.types.is_numeric_dtype(df[config.col_day]):
            df[config.col_day] = df[config.col_day].astype(str)
        try:
            df[config.col_day] = pd.to_datetime(df[config.col_day], format='%Y%m%d')
        except:
            df[config.col_day] = pd.to_datetime(df[config.col_day])

        all_cols = df.columns
        potential_feats = all_cols[config.feature_start_idx:].tolist()
        if config.col_node_idx in potential_feats: potential_feats.remove(config.col_node_idx)

        self.feature_cols = potential_feats
        self.num_features = len(self.feature_cols)
        self.num_nodes = df[config.col_node_idx].max() + 1

        unique_days = sorted(df[config.col_day].unique())
        self.day_map = {day: i for i, day in enumerate(unique_days)}
        self.idx_to_day = {i: day for i, day in enumerate(unique_days)}
        self.total_days = len(unique_days)
        self.idx_to_code = df.groupby(config.col_node_idx)[config.col_code].first().to_dict()

        print("Building Tensor...")
        self.global_features = torch.zeros((self.num_nodes, self.total_days, self.num_features), dtype=torch.float32)
        self.global_targets = torch.zeros((self.num_nodes, self.total_days), dtype=torch.float32)
        self.active_mask = torch.zeros((self.num_nodes, self.total_days), dtype=torch.bool)

        grouped = df.groupby(config.col_day)
        # 这里的 tqdm 可能会干扰主循环，建议在加载大数据时保留，或者设为 disable=True
        for day, group in tqdm(grouped, desc="Filling Data"):
            d_idx = self.day_map[day]
            node_idxs = group[config.col_node_idx].values
            feats = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)
            rets = torch.tensor(group[config.col_target].values, dtype=torch.float32)
            self.global_features[node_idxs, d_idx, :] = feats
            self.global_targets[node_idxs, d_idx] = rets
            self.active_mask[node_idxs, d_idx] = True

    def get_batch(self, day_idx, window):
        if day_idx < window: return None
        mask = self.active_mask[:, day_idx]
        active_indices = torch.where(mask)[0]
        if len(active_indices) == 0: return None
        start = day_idx - window + 1;
        end = day_idx + 1
        x = self.global_features[active_indices, start:end, :]
        y = self.global_targets[active_indices, day_idx]
        return x, y, active_indices


# ==========================================
# 4. 训练与评估流程 (封装函数)
# ==========================================
def ic_loss_fn(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return 1 - corr


def calc_ic(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return corr.item()


def run_single_model(model_name, dataset):
    """
    运行单个模型的完整训练和评估流程
    """
    # 1. 设置当前模型参数
    Config.model_type = model_name
    Config.project_name = f"Baseline_{model_name}"

    # 2. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(Config.base_output_dir, f"{timestamp}_{Config.project_name}")
    os.makedirs(run_dir, exist_ok=True)

    # 3. 设置日志 (保存原始 stdout 以便恢复)
    original_stdout = sys.stdout
    logger = Logger(os.path.join(run_dir, 'train_log.txt'), original_stdout)
    sys.stdout = logger

    print(f"\n{'#' * 60}")
    print(f"### 开始训练模型: {model_name}")
    print(f"### 输出目录: {run_dir}")
    print(f"{'#' * 60}\n")

    model = None  # 预声明
    try:
        # 4. 初始化模型
        input_dim = dataset.num_features
        if model_name == "MLP":
            model = BaselineMLP(input_dim, Config).to(Config.device)
        elif model_name == "LSTM":
            model = BaselineLSTM(input_dim, Config).to(Config.device)
        elif model_name == "GRU":
            model = BaselineGRU(input_dim, Config).to(Config.device)
        elif model_name == "Transformer":
            model = BaselineTransformer(input_dim, Config).to(Config.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

        # 5. 训练循环
        start_idx = Config.window_size
        train_end = Config.train_days

        best_ic = -999.0

        # 使用 original_stdout 打印进度条，避免写入日志文件导致文件过大
        for epoch in range(Config.num_epochs):
            model.train()
            loss_list = []

            pbar = tqdm(range(start_idx, train_end), desc=f"[{model_name}] Epoch {epoch + 1}", leave=False,
                        file=original_stdout)
            for day_idx in pbar:
                batch = dataset.get_batch(day_idx, Config.window_size)
                if batch is None: continue
                x, y, idx = batch
                x, y = x.to(Config.device), y.to(Config.device)

                optimizer.zero_grad()
                pred = model(x)

                l_mse = nn.functional.mse_loss(pred, y)
                l_ic = ic_loss_fn(pred, y)
                loss = (1 - Config.ic_loss_weight) * l_mse + Config.ic_loss_weight * l_ic

                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            # Valid
            model.eval()
            val_ics = []
            with torch.no_grad():
                for day_idx in range(train_end, train_end + Config.valid_days):
                    batch = dataset.get_batch(day_idx, Config.window_size)
                    if batch is None: continue
                    x, y, idx = batch
                    x, y = x.to(Config.device), y.to(Config.device)
                    pred = model(x)
                    val_ics.append(calc_ic(pred, y))

            avg_loss = np.mean(loss_list)
            avg_val_ic = np.mean(val_ics) if val_ics else 0

            print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | Valid IC: {avg_val_ic:.4f}")

            if avg_val_ic > best_ic:
                best_ic = avg_val_ic
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                print(f"  >>> New Best {model_name} Saved (IC: {best_ic:.4f})")

        # 6. 评估循环 (包含 Validation + Test)
        print(f"\n[{model_name}] Evaluation Phase...")
        if os.path.exists(os.path.join(run_dir, "best_model.pth")):
            model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth")))
        model.eval()

        test_start = Config.train_days
        test_end = dataset.total_days

        results = []
        with torch.no_grad():
            for day_idx in tqdm(range(test_start, test_end), desc="Inference", file=original_stdout):
                batch = dataset.get_batch(day_idx, Config.window_size)
                if batch is None: continue
                x, y, idx = batch
                x, y = x.to(Config.device), y.to(Config.device)
                pred = model(x)

                p_np = pred.cpu().numpy()
                y_np = y.cpu().numpy()
                curr_day = dataset.idx_to_day[day_idx]
                results.extend([[curr_day, p, t] for p, t in zip(p_np, y_np)])

        df_res = pd.DataFrame(results, columns=['day', 'pred', 'true'])

        # 统计指标
        daily_stats = []
        for day, g in df_res.groupby('day'):
            if len(g) < 10: continue
            ic, _ = pearsonr(g['pred'], g['true'])
            rank_ic, _ = spearmanr(g['pred'], g['true'])
            daily_stats.append({'day': day, 'IC': ic, 'RankIC': rank_ic})

        df_metrics = pd.DataFrame(daily_stats)
        df_metrics['day'] = pd.to_datetime(df_metrics['day'])
        df_metrics = df_metrics.sort_values('day')

        mean_rank_ic = df_metrics['RankIC'].mean()
        rank_ic_ir = mean_rank_ic / df_metrics['RankIC'].std()

        print("-" * 40)
        print(f"Model: {model_name} Final Results")
        print(f"Mean RankIC: {mean_rank_ic:.4f}")
        print(f"RankIC IR  : {rank_ic_ir:.4f}")
        print("-" * 40)

        # 简单绘图
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['day'], df_metrics['RankIC'].cumsum(), label=f'{model_name} Cumulative RankIC')
        plt.title(f"{model_name} Performance")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "cumulative_ic.png"))
        plt.close()

    except Exception as e:
        print(f"Error executing {model_name}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 7. 恢复控制台输出 & 关闭日志
        sys.stdout = original_stdout
        logger.close()

        # 8. 资源清理 (防止 OOM)
        if model is not None: del model
        if 'optimizer' in locals(): del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Finished {model_name}. Resources cleared.\n")


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 预加载数据 (只加载一次)
    print("Pre-loading dataset for all models...")
    dataset = TimeSeriesDataset(Config)

    # 2. 定义要跑的模型列表
    # 建议顺序：先跑小的，再跑大的
    models_to_train = ["MLP", "GRU", "LSTM", "Transformer"]

    # 3. 循环执行
    for model_name in models_to_train:
        run_single_model(model_name, dataset)

    print("\n" + "=" * 50)
    print("All baselines training completed!")
    print("=" * 50)