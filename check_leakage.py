import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# ==========================================
# 配置区域
# ==========================================
class Config:
    data_path = "366FactorsData_standard.parquet"  # 你的数据文件

    col_code = 'code'
    col_day = 'day'
    col_target = 'return'  # 原始收益率列名
    col_node_idx = 'node_idx'
    feature_start_idx = 3  # 特征起始列索引

    # 阈值设置：超过这个值的特征被视为“极其可疑”
    leakage_threshold = 0.8  # 强泄露 (几乎等于答案)
    suspicious_threshold = 0.2  # 可疑 (在这个场景下，单因子IC>0.2通常都不正常)


# ==========================================
# 绘图设置
# ==========================================
def set_plot_style():
    # 设置中文字体
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=fonts[0] if sys.platform == 'win32' else fonts[-1])


def check_data_leakage():
    set_plot_style()
    print(f"=== 数据泄露侦探启动 ===")
    print(f"正在读取数据: {Config.data_path} ...")

    if not os.path.exists(Config.data_path):
        print(f"错误: 文件不存在 {Config.data_path}")
        return

    df = pd.read_parquet(Config.data_path)

    # 1. 简单的数据清洗
    if pd.api.types.is_numeric_dtype(df[Config.col_day]):
        df[Config.col_day] = df[Config.col_day].astype(str)

    # 2. 复刻训练时的数据对齐逻辑 (关键步骤！)
    print("正在模拟训练逻辑 (Target Shift -1)...")
    df = df.sort_values([Config.col_code, Config.col_day])

    # 创建训练时模型真正看到的 Label (t+1 收益)
    df['TRAIN_LABEL'] = df.groupby(Config.col_code)[Config.col_target].shift(-1)

    # 去除最后一天 (没有Label的)
    df_clean = df.dropna(subset=['TRAIN_LABEL'])

    print(f"有效样本数: {len(df_clean)}")

    # 3. 提取特征列
    all_cols = df_clean.columns
    # 排除非特征列
    exclude_cols = [Config.col_code, Config.col_day, Config.col_node_idx,
                    Config.col_target, 'TRAIN_LABEL']

    # 假设从 feature_start_idx 开始是特征，但也排除上面的列
    potential_feats = all_cols[Config.feature_start_idx:].tolist()
    feature_cols = [c for c in potential_feats if c not in exclude_cols]

    print(f"正在扫描 {len(feature_cols)} 个特征的相关性...")
    print("注意：这可能需要几十秒...")

    # 4. 计算相关性 (Vectorized)
    # corrwith 可以快速计算 DataFrame 所有列与某一列的相关性
    correlations = df_clean[feature_cols].corrwith(df_clean['TRAIN_LABEL'])

    # 转为 DataFrame 并排序
    corr_df = correlations.to_frame(name='correlation')
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_corr', ascending=False)

    # 5. 打印报告
    print("\n" + "=" * 50)
    print(">>> 泄露检测报告 (Top 20 Suspicious Features) <<<")
    print("=" * 50)
    print(f"{'Feature Name':<30} | {'Correlation':<12} | {'Status'}")
    print("-" * 60)

    suspicious_list = []

    for feat_name, row in corr_df.head(20).iterrows():
        corr_val = row['correlation']
        abs_val = row['abs_corr']

        status = "正常"
        if abs_val > Config.leakage_threshold:
            status = "🚨 严重泄露!"
            suspicious_list.append(feat_name)
        elif abs_val > Config.suspicious_threshold:
            status = "⚠️ 高度可疑"

        print(f"{feat_name:<30} | {corr_val: .6f}    | {status}")

    print("-" * 60)

    if len(suspicious_list) > 0:
        print(f"\n[结论] 找到 {len(suspicious_list)} 个特征与 Label 高度相关！")
        print(f"建议立即从数据集中删除以下特征: \n{suspicious_list}")
    else:
        print(f"\n[结论] 未发现相关性 > {Config.leakage_threshold} 的明显泄露特征。")
        print("如果回测依然异常，请检查是否在数据预处理阶段使用了全局未来信息（如全局归一化）。")

    # 6. 绘图
    plt.figure(figsize=(12, 8))
    # 取绝对值最高的前30个画图
    top_30 = corr_df.head(30)

    sns.barplot(x=top_30['abs_corr'], y=top_30.index, palette='viridis')
    plt.axvline(Config.leakage_threshold, color='r', linestyle='--', label='Leakage Threshold')
    plt.title(f"Feature-Label Correlation (Top 30 Abs)\nTarget: Next Day Return")
    plt.xlabel("Absolute Pearson Correlation")
    plt.tight_layout()
    plt.savefig("leakage_check.png")
    print(f"可视化图表已保存: leakage_check.png")


if __name__ == "__main__":
    check_data_leakage()