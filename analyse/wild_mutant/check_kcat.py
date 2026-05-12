import pandas as pd
import os
from pathlib import Path
import numpy as np


def filter_kcat_dataset(input_path, output_dir, N=20):
    """
    筛选kcat数据集中突变体数量≥N的反应（UniProtID + Smiles 唯一组合）

    参数:
        input_path (str): 输入数据文件路径（支持.pkl/.csv格式）
        output_dir (str): 输出文件保存目录
        N (int): 筛选阈值（突变体数量下限，默认20，论文中也常用30）

    返回:
        pd.DataFrame: 筛选后的数据集
    """
    # ===================== 1. 输入校验与路径处理 =====================
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ===================== 2. 读取数据集 =====================
    print(f"正在读取数据集：{input_path}")
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == ".pkl":
        df = pd.read_pickle(input_path)
    elif file_ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"不支持的文件格式：{file_ext}，仅支持.pkl/.csv")

    # 检查必要列是否存在
    required_cols = ["UniProtID", "Smiles", "EnzymeType"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"数据集缺少必要列：{missing_cols}，请检查数据格式")

    # ===================== 3. 核心筛选逻辑 =====================
    df["reaction_id"] = df["UniProtID"] + "_" + df["Smiles"]
    reaction_mutant_count = df.groupby("reaction_id")["EnzymeType"].count()
    valid_reactions = reaction_mutant_count[reaction_mutant_count >= N].index.tolist()
    df_filtered = df[df["reaction_id"].isin(valid_reactions)].copy()
    df_filtered["is_mutant"] = df_filtered["EnzymeType"].apply(lambda x: 1 if x != "wild" else 0)

    # ===================== 4. 输出统计信息 =====================
    print(f"\n=== 突变体筛选结果（N={N}）===")
    print(f"原始数据集总样本数：{len(df):,}")
    print(f"筛选后总样本数：{len(df_filtered):,}")
    print(f"有效反应数（突变体数≥{N}）：{len(valid_reactions)}")
    print(f"筛选后突变体样本数：{df_filtered['is_mutant'].sum():,}")
    print(f"筛选后野生型样本数：{len(df_filtered) - df_filtered['is_mutant'].sum():,}")
    print(f"筛选保留率：{len(df_filtered) / len(df) * 100:.2f}%")

    # ===================== 5. 保存筛选后的数据（关键修改：保存为pkl） =====================
    # 保存为pickle格式，保留numpy数组原生格式
    output_filename = f"kcat-data_filtered_N{N}.pkl"
    output_path = os.path.join(output_dir, output_filename)
    df_filtered.to_pickle(output_path)  # 替换to_csv为to_pickle
    print(f"\n筛选后的数据已保存至：{output_path}")

    # 可选：额外保存CSV（仅用于查看，不用于预测）
    csv_filename = f"kcat-data_filtered_N{N}_view.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    # 只保存非数组列到CSV（避免字符串化）
    non_array_cols = [col for col in df_filtered.columns if not isinstance(df_filtered[col].iloc[0], np.ndarray)]
    df_filtered[non_array_cols].to_csv(csv_path, index=False, encoding="utf-8")
    print(f"CSV格式预览文件已保存至：{csv_path}（仅含非数组列）")

    return df_filtered


# ===================== 主程序调用 =====================
if __name__ == "__main__":
    INPUT_FILE = r"/mnt/usb3/wmx/kcat/kcat-data_feats_complete.pkl"
    OUTPUT_DIR = r"/mnt/usb3/wmx/analyse"
    FILTER_THRESHOLD = 20  # 可改为20

    try:
        filtered_df = filter_kcat_dataset(INPUT_FILE, OUTPUT_DIR, FILTER_THRESHOLD)
    except Exception as e:
        print(f"程序执行出错：{e}")