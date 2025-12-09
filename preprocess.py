import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ================= 配置 =================
RAW_DATA_PATH = "366FactorsData.parquet"  # 原始数据路径
SAVE_DATA_PATH = "366FactorsData_standard.parquet"  # 处理后保存的路径

COL_CODE = 'code'
COL_DAY = 'day'
COL_TARGET = 'return'
FEATURE_START_IDX = 3  # 因子从第4列开始


def preprocess_and_save():
    print(f"1. 正在读取原始数据: {RAW_DATA_PATH} ...")
    if not os.path.exists(RAW_DATA_PATH):
        print("错误：找不到原始数据文件！")
        return

    df = pd.read_parquet(RAW_DATA_PATH)
    print(f"   原始形状: {df.shape}")

    # 获取因子列
    feature_cols = df.columns[FEATURE_START_IDX:]
    print(f"   检测到 {len(feature_cols)} 个因子列")

    # ----------------------------------------------------
    # 2. 按天截面标准化 (Daily Z-Score) - 最耗时的一步
    # ----------------------------------------------------
    print("2. 执行按天截面标准化 (Daily Z-Score)...")

    # 定义标准化函数 (防止除零)
    def zscore_func(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    # 使用 groupby transform
    # 注意：这里直接修改原 DataFrame 可能会报警，但为了省内存我们直接赋值
    df[feature_cols] = df.groupby(COL_DAY)[feature_cols].transform(zscore_func)

    # 强制转为 float32 (减小体积，适配 PyTorch)
    df[feature_cols] = df[feature_cols].astype(np.float32)

    # ----------------------------------------------------
    # 3. 生成股票整数 ID (Node Index)
    # ----------------------------------------------------
    print("3. 生成全局股票代码整数索引 (node_idx)...")
    unique_codes = df[COL_CODE].unique()
    # 建立映射: code -> int
    code2idx = {code: i for i, code in enumerate(unique_codes)}

    # 映射并转为 int64
    df['node_idx'] = df[COL_CODE].map(code2idx).astype(np.int64)

    print(f"   全局股票数量: {len(unique_codes)}")

    # ----------------------------------------------------
    # 4. 保存处理后的数据
    # ----------------------------------------------------
    print(f"4. 正在保存至 {SAVE_DATA_PATH} ...")
    # 整理内存碎片
    df = df.copy()
    df.to_parquet(SAVE_DATA_PATH, index=False)

    print("\n[完成] 预处理结束！请在训练脚本中使用新生成的文件。")


if __name__ == "__main__":
    preprocess_and_save()