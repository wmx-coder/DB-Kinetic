import pickle
import numpy as np
import torch
import os
import gc
from tqdm import tqdm

# ========== 核心配置 ==========
RAW_PATHS = {
    "KCAT": "/mnt/usb1/wmx/eitlem/kcat_v1_with_fold_processed.pkl",
    "KM": "/mnt/usb1/wmx/eitlem/km_v1_with_fold_processed.pkl",
    "KKM": "/mnt/usb1/wmx/eitlem/kcat_km_with_log_feats_processed.pkl"
}
EITLEM_ROOT = "../data/EitlemData/"
FOLD_RANGE = range(0, 10)
BATCH_SIZE = 1000  # 分批处理的批次大小

# ========== 目录初始化 ==========
os.makedirs(os.path.join(EITLEM_ROOT, "PairInfo"), exist_ok=True)
os.makedirs(os.path.join(EITLEM_ROOT, "Feature/SmilesDict"), exist_ok=True)
# 注意：不再需要 Feature/ESMEmb 下的子目录，但保留父目录以防其他用途
os.makedirs(os.path.join(EITLEM_ROOT, "Feature/ESMEmb"), exist_ok=True)

# ========== 数据集改造核心逻辑 ==========
for dataset_type, raw_pkl in RAW_PATHS.items():
    print(f"\n===== 处理 {dataset_type} 数据集 =====")

    # 1. 加载原始数据（强制清理之前的内存）
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with open(raw_pkl, 'rb') as f:
        df = pickle.load(f)
    df["fold"] = df["fold"].astype(int)
    total_samples = len(df)
    print(f"✅ 加载原始数据：{total_samples}条样本")

    # 2. 分批生成全局唯一底物字典（降低内存峰值）
    # 2.1 分批计算底物哈希值
    df["sbt_feat_hash"] = None
    for start in range(0, total_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_samples)
        df.iloc[start:end, df.columns.get_loc("sbt_feat_hash")] = df.iloc[start:end]["sbt_feat"].apply(
            lambda x: hash(bytes(np.array(x).astype(np.float32)))
        )
        gc.collect()
    # 2.2 获取唯一哈希
    unique_hashes = list(set(df["sbt_feat_hash"].tolist()))
    print(f"✅ 去重后唯一底物数量：{len(unique_hashes)}")
    # 2.3 分批构建字典
    smiles_dict = {}
    for idx, hash_val in enumerate(tqdm(unique_hashes, desc="分批生成底物字典")):
        row = df[df["sbt_feat_hash"] == hash_val].iloc[0]
        sbt_feat_935 = np.array(row["sbt_feat"])
        maccs_167 = torch.FloatTensor(sbt_feat_935[-167:].astype(np.float32))
        smiles_dict[idx] = maccs_167
        if idx % BATCH_SIZE == 0:
            gc.collect()
    # 2.4 保存字典
    dict_save_path = os.path.join(EITLEM_ROOT, f"Feature/SmilesDict/{dataset_type}_SmilesDict.pt")
    torch.save(smiles_dict, dict_save_path)
    print(
        f"✅ 底物字典保存完成：{len(smiles_dict)}个唯一底物，文件大小≈{os.path.getsize(dict_save_path) / 1024 / 1024:.2f}MB")
    # 2.5 分批映射sbt_id
    sbt_hash_to_id = {h: i for i, h in enumerate(unique_hashes)}
    df["sbt_id"] = None
    for start in range(0, total_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_samples)
        df.iloc[start:end, df.columns.get_loc("sbt_id")] = df.iloc[start:end]["sbt_feat_hash"].map(sbt_hash_to_id)
        gc.collect()

    # 3. 生成10折轻量索引文件（核心：使用原始值，精准匹配列名）
    if dataset_type == "KCAT":
        label_col = "kcat(s^-1)"
    elif dataset_type == "KM":
        label_col = "Km(M)"
    else:  # KKM
        print(f"⚠️ KKM数据集：计算原始kcat/Km值（kcat(s^-1)/Km(M)）")
        df["kcat_over_Km"] = df["kcat(s^-1)"] / df["Km(M)"]
        label_col = "kcat_over_Km"

    if label_col not in df.columns:
        raise ValueError(f"数据集{dataset_type}中未找到标签列{label_col}，请检查！")

    for fold_num in FOLD_RANGE:
        train_df = df[df["fold"] != fold_num].reset_index(drop=True)
        test_df = df[df["fold"] == fold_num].reset_index(drop=True)

        def gen_pair_info(sub_df):
            pair_info = []
            batch_count = 0
            for _, row in tqdm(sub_df.iterrows(), desc=f"生成Fold{fold_num}索引"):
                pair_info.append((row["esm_id"], row["sbt_id"], row[label_col]))
                batch_count += 1
                if batch_count % BATCH_SIZE == 0:
                    gc.collect()
            return pair_info

        train_info_path = os.path.join(EITLEM_ROOT, f"PairInfo/{dataset_type}_Fold{fold_num}_Train.pt")
        test_info_path = os.path.join(EITLEM_ROOT, f"PairInfo/{dataset_type}_Fold{fold_num}_Test.pt")
        torch.save(gen_pair_info(train_df), train_info_path)
        torch.save(gen_pair_info(test_df), test_info_path)
        print(f"  - Fold{fold_num}索引保存完成：训练集{len(train_df)}条，测试集{len(test_df)}条")
        del train_df, test_df
        gc.collect()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ✂️ 删除原“蛋白Embedding软链接”部分（共约10行）
    # 理由：ESM特征已统一存于全局目录 /fs1/home/zhaott/m/Data/Feature/esm2_t33_650M_UR50D_unique/
    # 训练时将直接读取该目录，无需软链接
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # 5. 强制清理内存
    del df, smiles_dict, sbt_hash_to_id, unique_hashes
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:
        os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
    except Exception as e:
        print(f"⚠️ 清理系统缓存失败：{e}")
    print(f"✅ {dataset_type}数据集处理完成，已清理内存\n")

print("\n===== 数据集改造全部完成！=====")
print(f"Eitlem格式数据根路径：{EITLEM_ROOT}")