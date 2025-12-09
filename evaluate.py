import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_1samp


# ==========================================
# 0. 工具类：日志记录器
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


# ==========================================
# 1. 全局绘图风格设置
# ==========================================
def set_plot_style():
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=fonts[0] if sys.platform == 'win32' else fonts[2])
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})


set_plot_style()


# ==========================================
# 2. 配置与定义
# ==========================================
class Config:
    project_name = "Temporal_AdaGNN_Eval"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "366FactorsData_standard.parquet"

    col_code = 'code'
    col_day = 'day'
    col_target = 'return'
    col_node_idx = 'node_idx'
    feature_start_idx = 3

    window_size = 20
    train_days = 600
    valid_days = 100

    hidden_dim = 64
    node_emb_dim = 32
    dropout = 0.5


# --- 模型定义 ---
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
        self.gru = nn.GRU(input_dim, config.hidden_dim, batch_first=True)
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
        h_self = self.bn(hn[-1])
        adj = self.graph_learner(active_indices)
        h_graph = self.gnn(h_self, adj)
        h_cat = torch.cat([h_self, h_graph], dim=1)
        return self.head(h_cat).squeeze()


class TimeSeriesDataset:
    def __init__(self, config):
        print(f"Loading Data: {config.data_path}")
        df = pd.read_parquet(config.data_path)

        # [修复 1] 强制处理日期格式
        # 如果是 20230101 这种整数，转为字符串 '20230101'，后续 Pandas 能更好地识别
        if pd.api.types.is_numeric_dtype(df[config.col_day]):
            print("检测到日期列为数值型，正在转换为字符串以确保解析正确...")
            df[config.col_day] = df[config.col_day].astype(str)

        # 也可以尝试直接转为 datetime 对象，确保后续逻辑统一
        try:
            # 尝试标准格式 YYYYMMDD
            df[config.col_day] = pd.to_datetime(df[config.col_day], format='%Y%m%d')
        except:
            try:
                # 尝试自动推断
                df[config.col_day] = pd.to_datetime(df[config.col_day])
            except Exception as e:
                print(f"警告：日期转换失败，将保持原样。错误: {e}")

        all_cols = df.columns
        potential_feats = all_cols[config.feature_start_idx:].tolist()
        if config.col_node_idx in potential_feats:
            potential_feats.remove(config.col_node_idx)

        self.feature_cols = potential_feats
        self.num_features = len(self.feature_cols)
        self.num_nodes = df[config.col_node_idx].max() + 1

        # 这里的 unique_days 现在是 Timestamp 对象了
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
        for day, group in tqdm(grouped, desc="Filling"):
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

        start = day_idx - window + 1
        end = day_idx + 1
        x = self.global_features[active_indices, start:end, :]
        y = self.global_targets[active_indices, day_idx]
        return x, y, active_indices


# ==========================================
# 3. 核心评估流程
# ==========================================
def run_evaluation(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    eval_dir = os.path.dirname(model_path)
    log_dir = os.path.join(eval_dir, "logs")
    result_dir = os.path.join(eval_dir, "results")

    for directory in [log_dir, result_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    log_file = os.path.join(log_dir, "evaluation_result.log")
    sys.stdout = Logger(log_file, sys.stdout)

    print(f"\n{'=' * 50}")
    print(f">>> 启动学术评估程序 (含回测) <<<")
    print(f"日志文件: {log_file}")
    print(f"目标模型: {model_path}")
    print(f"{'=' * 50}\n")

    # 1. 准备数据
    dataset = TimeSeriesDataset(Config)

    # 2. 加载模型
    model = TemporalAdaGNN(dataset.num_nodes, dataset.num_features, Config)
    try:
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
    except RuntimeError as e:
        print(f"\n[Error] 权重加载失败！{e}")
        return

    model.to(Config.device)
    model.eval()

    # 3. 确定测试区间
    test_start = Config.train_days
    test_end = dataset.total_days

    print(f"测试集区间: Index {test_start} -> {test_end} (共 {test_end - test_start} 天)")
    if test_start >= test_end:
        print("错误: 测试集为空")
        return

    # 4. 推理循环
    results = []
    with torch.no_grad():
        for day_idx in tqdm(range(test_start, test_end), desc="Predicting"):
            batch = dataset.get_batch(day_idx, Config.window_size)
            if batch is None: continue

            x, y, idx = batch
            x, y, idx = x.to(Config.device), y.to(Config.device), idx.to(Config.device)

            pred = model(x, idx)

            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            idx_np = idx.cpu().numpy()

            # 这里 dataset.idx_to_day[day_idx] 已经是 Timestamp 对象了
            curr_day = dataset.idx_to_day[day_idx]

            for i in range(len(idx_np)):
                code = dataset.idx_to_code.get(idx_np[i], str(idx_np[i]))
                results.append({
                    'day': curr_day,
                    'code': code,
                    'pred': float(pred_np[i]),
                    'true': float(y_np[i])
                })

    df_res = pd.DataFrame(results)
    print(f"预测完成，共生成 {len(df_res)} 条预测记录。")

    # ==========================================
    # 4. 统计指标计算
    # ==========================================
    print("\n[Step 1] 计算统计指标 (IC/RankIC)...")
    daily_metrics = []

    for day, group in df_res.groupby('day'):
        if len(group) < 10: continue

        p = group['pred'].values
        t = group['true'].values

        ic, _ = pearsonr(p, t)
        rank_ic, _ = spearmanr(p, t)
        mse = np.mean((p - t) ** 2)

        daily_metrics.append({
            'day': day,
            'IC': ic,
            'RankIC': rank_ic,
            'MSE': mse
        })

    df_metrics = pd.DataFrame(daily_metrics)

    # 确保日期列是 datetime 格式，方便绘图
    df_metrics['day'] = pd.to_datetime(df_metrics['day'])
    df_metrics = df_metrics.sort_values('day')

    mean_ic = df_metrics['IC'].mean()
    ic_std = df_metrics['IC'].std()
    icir = mean_ic / ic_std if ic_std != 0 else 0
    t_stat, p_val = ttest_1samp(df_metrics['RankIC'], 0)

    mean_rank_ic = df_metrics['RankIC'].mean()
    rank_ic_std = df_metrics['RankIC'].std()
    rank_icir = mean_rank_ic / rank_ic_std if rank_ic_std != 0 else 0

    print("-" * 50)
    print(">>> 学术回归评估报告 (Academic Report) <<<")
    print(f"MSE (Error)       : {df_metrics['MSE'].mean():.6f}")
    print("-" * 50)
    print(f"Mean IC           : {mean_ic:.4f}")
    print(f"ICIR              : {icir:.4f}")
    print("-" * 50)
    print(f"Mean RankIC       : {mean_rank_ic:.4f}")
    print(f"RankICIR          : {rank_icir:.4f}")
    print(f"RankIC t-stat     : {t_stat:.2f} (p-value: {p_val:.4e})")
    print("-" * 50)

    # ==========================================
    # 5. 分层回测
    # ==========================================
    print("\n[Step 2] 进行分层回测 (Top vs Bottom)...")
    df_res['group'] = df_res.groupby('day')['pred'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )

    group_ret = df_res.groupby(['day', 'group'])['true'].mean().unstack()
    cols = group_ret.columns

    if len(cols) >= 2:
        top_col = cols[-1]
        bot_col = cols[0]
        group_ret['Long-Short'] = group_ret[top_col] - group_ret[bot_col]

        ann_ret_ls = (1 + group_ret['Long-Short']).prod() ** (252 / len(group_ret)) - 1
        print(f"多空对冲年化收益率: {ann_ret_ls:.2%}")
    else:
        print("警告：分组数量不足，无法计算多空收益")
        group_ret['Long-Short'] = 0.0

    # 将索引确保为 DatetimeIndex
    group_ret.index = pd.to_datetime(group_ret.index)
    group_cum = group_ret.cumsum()

    # ==========================================
    # 6. 可视化绘图 (X轴修复版)
    # ==========================================
    print("\n[Step 3] 生成图表...")
    fig = plt.figure(figsize=(18, 10))

    # 提取日期序列用于绘图 (确保是 datetime 类型)
    plot_dates = df_metrics['day']
    backtest_dates = group_ret.index

    # 图 1: 累积 RankIC
    ax1 = fig.add_subplot(2, 2, 1)
    df_metrics['cumsum_RankIC'] = df_metrics['RankIC'].cumsum()

    # [修复] 使用正确的 dates 列表绘图
    ax1.plot(plot_dates, df_metrics['cumsum_RankIC'], label='Cumulative RankIC', color='blue')
    ax1.set_title(f"累积 RankIC\nMean: {mean_rank_ic:.4f}, IR: {rank_icir:.2f}")
    ax1.legend()
    ax1.grid(True)
    # 旋转日期标签防止重叠
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

    # 图 2: RankIC 分布
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(df_metrics['RankIC'], bins=40, color='green', alpha=0.6, edgecolor='black')
    ax2.axvline(mean_rank_ic, color='red', linestyle='--', label='Mean')
    ax2.set_title("RankIC 分布直方图")
    ax2.legend()

    # 图 3: 分组累计收益
    ax3 = fig.add_subplot(2, 2, 3)
    if len(cols) >= 1:
        if 4 in cols: ax3.plot(backtest_dates, group_cum[4], label='Top 20%', color='red')
        if 2 in cols: ax3.plot(backtest_dates, group_cum[2], label='Middle', color='grey', alpha=0.5)
        if 0 in cols: ax3.plot(backtest_dates, group_cum[0], label='Bottom 20%', color='green')
    ax3.set_title("分层累计收益 (单利)")
    ax3.legend()
    ax3.grid(True)
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

    # 图 4: 多空净值
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(backtest_dates, group_cum['Long-Short'], label='Long-Short', color='purple')
    ax4.set_title("多空对冲表现 (Alpha)")
    ax4.legend()
    ax4.grid(True)
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()

    # 保存文件
    df_res.to_csv(os.path.join(result_dir, "test_predictions_raw.csv"), index=False)
    df_metrics.to_csv(os.path.join(result_dir, "test_metrics_daily.csv"), index=False)
    group_ret.to_csv(os.path.join(result_dir, "backtest_returns.csv"))

    plot_path = os.path.join(log_dir, "academic_report.png")
    plt.savefig(plot_path, dpi=300)

    print(f"图表已保存: {plot_path}")
    print("评估完成。")


if __name__ == "__main__":
    target_model = r"E:\code\factorsforcast\runs\20251209_144100_Temporal_AdaGNN\best_model.pth"
    try:
        run_evaluation(target_model)
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()