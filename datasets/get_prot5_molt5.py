import pandas as pd
import numpy as np
import torch as th
import os
from utils_prot5_molt5 import Seq_to_vec, get_molT5_embed, GetMACCSKeys
from sklearn.model_selection import KFold  # 可选：如需fold划分
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
# ---------------------- 1. 配置路径（替换为你的实际路径） ----------------------
RAW_DATA_PATH = "/mnt/usb/code/wm/catapro/datasets/Km-data_0.4simi-10fold.csv"  # 你的原始CSV
ProtT5_model = "/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50"
MolT5_model = "/mnt/usb3/code/gfy/code/CataPro-master/models/molt5-base-smiles2caption"
OUTPUT_PATH = "/mnt/usb/code/wm/catapro/datasets/km_data/Km-data_0.4simi-10fold.pkl"  # 最终Pickle文件


# ---------------------- 2. 读取原始数据（完整保留所有列） ----------------------
def load_raw_data(raw_path):
    """读取原始CSV，保留所有列，仅清洗空值"""
    df = pd.read_csv(raw_path)
    # 核心列检查（确保有序列、SMILES、kcat，其他列保留）
    required_cols = ["Sequence", "Smiles", "Km(M)"]  # 你的原始列名，需与实际匹配
    assert all(col in df.columns for col in required_cols), "原始数据缺少核心列（sequence/smiles/Km）！"

    # 清洗：仅删除核心列有空值的样本（其他列空值可保留）
    df_clean = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"原始样本数：{len(df)} → 清洗后有效样本数：{len(df_clean)}")
    return df_clean


# ---------------------- 3. 提取特征（返回数组类型，不拆分列） ----------------------
# ---------------------- 3. 提取特征（分批处理） ----------------------
def extract_features(df_clean, prot5_path, molt5_path, batch_size=16, save_every=2000, output_dir="./features_cache"):
    os.makedirs(output_dir, exist_ok=True)

    ezy_feats, sbt_feats = [], []
    n = len(df_clean)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        print(f"处理 {start} ~ {end} / {n}")

        # ProtT5
        batch_seq = df_clean["Sequence"].iloc[start:end].tolist()
        batch_ezy = Seq_to_vec(batch_seq, ProtT5_model=prot5_path)
        ezy_feats.extend(batch_ezy)

        # MolT5 + MACCS
        batch_smiles = df_clean["Smiles"].iloc[start:end].tolist()
        batch_molt5 = get_molT5_embed(batch_smiles, molt5_path)
        batch_maccs = GetMACCSKeys(batch_smiles)
        batch_sbt = np.concatenate([batch_molt5, batch_maccs], axis=1)
        sbt_feats.extend(batch_sbt)

        # 定期存储到文件，避免崩掉
        if start % save_every == 0 and start > 0:
            np.save(os.path.join(output_dir, f"ezy_feats_{start}_{end}.npy"), np.array(ezy_feats))
            np.save(os.path.join(output_dir, f"sbt_feats_{start}_{end}.npy"), np.array(sbt_feats))
            print(f"已保存临时文件：{start}-{end}")

    return ezy_feats, sbt_feats



# ---------------------- 4. 新增log(kcat)列（处理异常值） ----------------------
def add_log_km(df_clean):
    """对kcat(s^-1)做对数转换，处理kcat≤0的情况（避免log报错）"""
    kcat_raw = df_clean["Km(M)"].values
    # 处理逻辑：kcat≤0时，用极小值（如1e-8）替换（避免log(0)或log负数）
    kcat_clean = np.where(kcat_raw > 0, kcat_raw, 1e-8)
    # 计算log10(kcat)（或log自然对数，根据模型需求选择，这里以log10为例）
    df_clean["log10_km"] = np.log10(kcat_clean)
    print("已新增log10_km列（模型训练用目标值）")
    # 可选：打印转换统计，确认无异常
    print(f"原始km范围：{kcat_raw.min():.4f} ~ {kcat_raw.max():.4f}")
    print(f"log10(km)范围：{df_clean['log10_km'].min():.4f} ~ {df_clean['log10_km'].max():.4f}")
    return df_clean


# ---------------------- 5. 主函数：整合所有步骤并保存 ----------------------
if __name__ == "__main__":
    # 步骤1：读取原始数据（保留所有列）
    df_full = load_raw_data(RAW_DATA_PATH)

    # 步骤2：提取酶特征和分子特征（返回数组列表）
    ezy_feats, sbt_feats = extract_features(
        df_clean=df_full,
        prot5_path=ProtT5_model,
        molt5_path=MolT5_model
    )

    # 步骤3：新增特征列（酶特征、分子特征）
    df_full["ezy_feat"] = ezy_feats  # 新增列：每个元素是1024维数组
    df_full["sbt_feat"] = sbt_feats  # 新增列：每个元素是935维数组

    # 步骤4：新增log(kcat)列（模型训练目标值）
    df_full = add_log_km(df_full)

    # （可选步骤5：新增fold划分列，用于10折训练）
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # df_full["fold"] = 0  # 初始化fold列
    # for fold_idx, (_, valid_idx) in enumerate(kf.split(df_full)):
    #     df_full.loc[valid_idx, "fold"] = fold_idx  # 给验证集分配fold号
    # print("已新增fold列（10折交叉验证用）")

    # 步骤6：保存为Pickle文件（完整保留所有列和数组类型）
    df_full.to_pickle(OUTPUT_PATH)
    print(f"\n文件已保存到：{OUTPUT_PATH}")
    print(f"最终文件包含列：{list(df_full.columns)}")
    print(f"样本数：{len(df_full)}，新增列：ezy_feat（酶特征）、sbt_feat（分子特征）、log10_km（对数km）、fold（可选）")


# ---------------------- 验证：读取Pickle文件确认格式 ----------------------
def verify_pickle(file_path):
    df_verify = pd.read_pickle(file_path)
    print("\n=== 验证Pickle文件 ===")
    print(f"文件列名：{list(df_verify.columns)}")
    print(f"第1个样本的酶特征形状：{df_verify['ezy_feat'].iloc[0].shape} → 应为(1024,)")
    print(f"第1个样本的分子特征形状：{df_verify['sbt_feat'].iloc[0].shape} → 应为(935,)")
    print(f"第1个样本的log10_km值：{df_verify['log10_km'].iloc[0]:.4f}")
    if "fold" in df_verify.columns:
        print(f"第1个样本的fold值：{df_verify['fold'].iloc[0]}")
    else:
        print("当前数据没有 fold 列")



# 运行验证
verify_pickle(OUTPUT_PATH)