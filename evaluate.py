import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ==========================================
# 0. 配置与模型定义 (需与训练脚本一致)
# ==========================================
class Config:
    # 基础配置
    project_name = "Temporal_AdaGNN"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "366FactorsData_standard.parquet"

    # 列名
    col_code = 'code'
    col_day = 'day'
    col_target = 'return'
    col_node_idx = 'node_idx'
    feature_start_idx = 3

    # 参数
    window_size = 20
    train_days = 600
    valid_days = 100

    hidden_dim = 64
    node_emb_dim = 32
    dropout = 0.5
    gnn_layers = 2

    # 回测特有参数
    fee_rate = 0.0015  # 双边万三+印花税 -> 约千1.5
    turnover_rate = 0.5  # 预估日换手率


# --- 模型类定义 (复制自训练脚本，确保结构一致) ---
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
        row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        return adj / row_sum


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
        self.gru = nn.GRU(input_dim, config.hidden_dim, batch_first=True, num_layers=1)
        self.bn = nn.BatchNorm1d(config.hidden_dim)
        self.graph_learner = AdaptiveGraphLearner(num_nodes, config.node_emb_dim)
        self.gnn = GNNLayer(config.hidden_dim, config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, active_indices):
        self.gru.flatten_parameters()
        out, hn = self.gru(x)
        h_temporal = self.bn(hn[-1])
        adj = self.graph_learner(active_indices)
        h_spatial = self.gnn(h_temporal, adj)
        h_final = torch.cat([h_temporal, h_spatial], dim=1)
        return self.head(h_final).squeeze()


class TimeSeriesDataset:
    def __init__(self, config):
        print(f"[Eval Data] 加载: {config.data_path}")
        df = pd.read_parquet(config.data_path)

        # 1. 日期处理
        if pd.api.types.is_numeric_dtype(df[config.col_day]):
            df[config.col_day] = df[config.col_day].astype(str)
        try:
            df[config.col_day] = pd.to_datetime(df[config.col_day], format='%Y%m%d')
        except:
            df[config.col_day] = pd.to_datetime(df[config.col_day])

        # 2. 收益率对齐 (必须与训练一致: Shift -1)
        print("[Eval Data] 对齐未来收益率...")
        df = df.sort_values([config.col_code, config.col_day])
        df['target_y'] = df.groupby(config.col_code)[config.col_target].shift(-1)
        df = df.dropna(subset=['target_y'])

        # 3. 特征列
        all_cols = df.columns
        potential_feats = all_cols[config.feature_start_idx:].tolist()
        exclude_cols = [config.col_node_idx, 'target_y', config.col_target, config.col_code, config.col_day]
        self.feature_cols = [c for c in potential_feats if c not in exclude_cols]
        self.num_features = len(self.feature_cols)
        self.num_nodes = df[config.col_node_idx].max() + 1

        # 4. 映射
        unique_days = sorted(df[config.col_day].unique())
        self.day_map = {day: i for i, day in enumerate(unique_days)}
        self.idx_to_day = {i: day for i, day in enumerate(unique_days)}
        self.idx_to_code = df.groupby(config.col_node_idx)[config.col_code].first().to_dict()
        self.total_days = len(unique_days)

        # 5. Tensor
        print("[Eval Data] 构建张量...")
        self.global_features = torch.zeros((self.num_nodes, self.total_days, self.num_features), dtype=torch.float32)
        self.global_targets = torch.zeros((self.num_nodes, self.total_days), dtype=torch.float32)
        self.active_mask = torch.zeros((self.num_nodes, self.total_days), dtype=torch.bool)

        grouped = df.groupby(config.col_day)
        for day, group in tqdm(grouped, desc="Filling"):
            d_idx = self.day_map[day]
            node_idxs = group[config.col_node_idx].values
            feats = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)
            rets = torch.tensor(group['target_y'].values, dtype=torch.float32)

            self.global_features[node_idxs, d_idx, :] = feats
            self.global_targets[node_idxs, d_idx] = rets
            self.active_mask[node_idxs, d_idx] = True

    def get_batch(self, day_idx, window):
        if day_idx < window: return None
        mask = self.active_mask[:, day_idx]
        active_indices = torch.where(mask)[0]
        if len(active_indices) == 0: return None
        start = day_idx - window + 1
        end = day_idx + 1
        x = self.global_features[active_indices, start:end, :]
        y = self.global_targets[active_indices, day_idx]
        return x, y, active_indices


# ==========================================
# 1. 工具类
# ==========================================
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_plot_style():
    # 设置中文字体，防止乱码
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=fonts[0] if sys.platform == 'win32' else fonts[-1])


# ==========================================
# 2. 核心评估函数
# ==========================================
def run_evaluation(model_path):
    set_plot_style()

    if not os.path.exists(model_path):
        print(f"文件不存在: {model_path}")
        return

    # 设置输出目录
    eval_dir = os.path.dirname(os.path.dirname(model_path))  # 回退到 run_dir
    log_dir = os.path.join(eval_dir, "eval_logs")
    res_dir = os.path.join(eval_dir, "eval_results")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'evaluation.log'), sys.stdout)
    print(f"=== Evaluation Started: {model_path} ===")

    # 1. 加载数据
    dataset = TimeSeriesDataset(Config)

    # 2. 加载模型
    model = TemporalAdaGNN(dataset.num_nodes, dataset.num_features, Config)
    # 安全加载
    try:
        # weights_only=False 兼容旧版 pytorch
        state = torch.load(model_path, map_location=Config.device)
        model.load_state_dict(state)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    model.to(Config.device)
    model.eval()

    # 3. 推理 (Test Set)
    test_start = Config.train_days
    test_end = dataset.total_days
    print(f"测试区间: {test_start} -> {test_end}")

    results = []
    with torch.no_grad():
        for day_idx in tqdm(range(test_start, test_end), desc="Predicting"):
            batch = dataset.get_batch(day_idx, Config.window_size)
            if batch is None: continue

            x, y, idx = batch
            x, y, idx = x.to(Config.device), y.to(Config.device), idx.to(Config.device)

            pred = model(x, idx)

            # 收集结果
            p_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            idx_np = idx.cpu().numpy()
            curr_day = dataset.idx_to_day[day_idx]

            for i in range(len(idx_np)):
                code = dataset.idx_to_code.get(idx_np[i], str(idx_np[i]))
                results.append({
                    'day': curr_day,
                    'code': code,
                    'pred': float(p_np[i]),
                    'true': float(y_np[i])
                })

    df_res = pd.DataFrame(results)
    df_res['day'] = pd.to_datetime(df_res['day'])

    # ==========================================
    # 4. 统计指标计算
    # ==========================================
    daily_metrics = []
    for day, group in df_res.groupby('day'):
        if len(group) < 10: continue
        ic, _ = pearsonr(group['pred'], group['true'])
        rank_ic, _ = spearmanr(group['pred'], group['true'])
        daily_metrics.append({'day': day, 'IC': ic, 'RankIC': rank_ic})

    df_metrics = pd.DataFrame(daily_metrics).sort_values('day')
    mean_rank_ic = df_metrics['RankIC'].mean()
    rank_icir = mean_rank_ic / df_metrics['RankIC'].std()

    # ==========================================
    # 5. 回测与绩效计算
    # ==========================================
    # 分组 (5组)
    df_res['group'] = df_res.groupby('day')['pred'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )

    # 原始日收益
    group_ret = df_res.groupby(['day', 'group'])['true'].mean().unstack().fillna(0)

    # 扣费 (日固定拖累)
    cost = Config.turnover_rate * (Config.fee_rate * 2)
    group_ret_net = group_ret - cost

    # 多空
    cols = group_ret.columns
    top_col = cols[-1]
    bot_col = cols[0]
    group_ret_net['Long-Short'] = group_ret_net[top_col] - group_ret_net[bot_col]

    # 净值 (单利 1.0 起点)
    nav = 1.0 + group_ret_net.cumsum()

    # --- 核心绩效函数 ---
    def calc_metrics(ret, nav_series):
        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Max Drawdown
        roll_max = nav_series.cummax()
        dd = (nav_series - roll_max) / roll_max
        mdd = dd.min()

        calmar = abs(ann_ret / mdd) if mdd != 0 else 0
        win_rate = len(ret[ret > 0]) / len(ret)

        return [ann_ret, ann_vol, sharpe, mdd, calmar, win_rate]

    m_top = calc_metrics(group_ret_net[top_col], nav[top_col])
    m_ls = calc_metrics(group_ret_net['Long-Short'], nav['Long-Short'])

    # 打印报表
    print("-" * 65)
    print(f"{'>>> Deep Backtest Report <<<':^65}")
    print("-" * 65)
    print(f"{'Metrics':<20} | {'Top 20% (Long)':^18} | {'Long-Short':^18}")
    print("-" * 65)
    labels = ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Win Rate"]
    fmts = ["{:.2%}", "{:.2%}", "{:.4f}", "{:.2%}", "{:.4f}", "{:.2%}"]

    for i in range(len(labels)):
        print(f"{labels[i]:<20} | {fmts[i].format(m_top[i]):^18} | {fmts[i].format(m_ls[i]):^18}")

    print("-" * 65)
    print(f"Mean RankIC: {mean_rank_ic:.4f} | RankICIR: {rank_icir:.4f}")
    print("-" * 65)

    # ==========================================
    # 6. 绘图
    # ==========================================
    fig = plt.figure(figsize=(18, 10))

    # Subplot 1: Cumulative RankIC
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df_metrics['day'], df_metrics['RankIC'].cumsum(), label='Cum RankIC', color='#1f77b4')
    ax1.set_title(f"Cumulative RankIC (Mean={mean_rank_ic:.3f})")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Drawdown Area
    ax2 = fig.add_subplot(2, 2, 2)
    nav_series = nav[top_col]
    dd = (nav_series - nav_series.cummax()) / nav_series.cummax()
    ax2.fill_between(nav.index, dd, 0, color='red', alpha=0.3)
    ax2.set_title(f"Top 20% Drawdown (MDD={m_top[3]:.2%})")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Layered NAV
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(nav.index, nav[top_col], label='Top 20%', color='#d62728')
    ax3.plot(nav.index, nav[bot_col], label='Bottom 20%', color='#2ca02c')
    ax3.set_title("Layered NAV (Net of Fee)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Long-Short NAV
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(nav.index, nav['Long-Short'], label='Long-Short', color='#9467bd')
    ax4.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title("Long-Short Hedge NAV")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, "backtest_report.png")
    plt.savefig(plot_path, dpi=300)
    print(f"图表已保存: {plot_path}")

    # 保存数据
    df_res.to_csv(os.path.join(res_dir, "pred_detail.csv"), index=False)
    nav.to_csv(os.path.join(res_dir, "nav_history.csv"))


if __name__ == "__main__":
    # 请修改这里的路径为你实际训练生成的 best_model.pth 路径
    target_model = r"E:\code\TemporalAdaGNN\runs\20251215_191607_Temporal_AdaGNN\best_model.pth"

    run_evaluation(target_model)