import pickle
import numpy as np
import torch
import os
import gc
from tqdm import tqdm

# ========== 核心配置 ==========
RAW_PATHS = {
    "KCAT": "/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete_esm_id.pkl",
    "KM": "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats_with_esm_id.pkl",
    "KKM": "/mnt/usb3/code/wm/data/kcat_km/kcat_km_v1_with_fold_esm_id.pkl"
}
EITLEM_ROOT = "/mnt/usb3/code/wm/data/EitlemData/"
FOLD_RANGE = range(0, 10)
BATCH_SIZE = 1000  # 分批处理的批次大小

# ========== 目录初始化 ==========
os.makedirs(os.path.join(EITLEM_ROOT, "PairInfo"), exist_ok=True)
os.makedirs(os.path.join(EITLEM_ROOT, "Feature/SmilesDict"), exist_ok=True)
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
        # smiles_string = row["Smiles"]
        # smiles_dict[idx] = smiles_string
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
        # KCAT：直接使用原始列 kcat(s^-1)
        label_col = "kcat(s^-1)"
    elif dataset_type == "KM":
        # KM：直接使用原始列 Km(M)
        label_col = "Km(M)"
    else:  # KKM
        # KKM：无直接原始列，计算 kcat(s^-1)/Km(M) 作为原始值
        print(f"⚠️ KKM数据集：计算原始kcat/Km值（kcat(s^-1)/Km(M)）")
        df["kcat_over_Km"] = df["kcat(s^-1)"] / df["Km(M)"]
        label_col = "kcat_over_Km"

    # 验证标签列存在性
    if label_col not in df.columns:
        raise ValueError(f"数据集{dataset_type}中未找到标签列{label_col}，请检查！")

    for fold_num in FOLD_RANGE:
        train_df = df[df["fold"] != fold_num].reset_index(drop=True)
        test_df = df[df["fold"] == fold_num].reset_index(drop=True)


        def gen_pair_info(sub_df):
            pair_info = []
            batch_count = 0
            for _, row in tqdm(sub_df.iterrows(), desc=f"生成Fold{fold_num}索引"):
                # 核心：使用原始标签值
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
        # 清理临时df
        del train_df, test_df
        gc.collect()

    # 4. 蛋白Embedding软链接
    dataset_type_lower = dataset_type.lower() if dataset_type != "KKM" else "kcat_km"
    esm_src_root = f"/mnt/usb3/code/wm/esm/{dataset_type_lower}/"
    esm_dst_root = os.path.join(EITLEM_ROOT, f"Feature/ESMEmb/{dataset_type}/")
    os.makedirs(esm_dst_root, exist_ok=True)

    for esm_id in tqdm(df["esm_id"].unique(), desc="创建ESM软链接"):
        src_file = os.path.join(esm_src_root, f"unique_{dataset_type_lower}_{esm_id}.pt")
        dst_file = os.path.join(esm_dst_root, f"{esm_id}.pt")
        if os.path.exists(src_file) and not os.path.exists(dst_file):
            os.symlink(src_file, dst_file)

    # 5. 强制清理内存
    del df, smiles_dict, sbt_hash_to_id, unique_hashes
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # 清理系统缓存（可选）
    try:
        os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
    except Exception as e:
        print(f"⚠️ 清理系统缓存失败：{e}")
    print(f"✅ {dataset_type}数据集处理完成，已清理内存\n")

print("\n===== 数据集改造全部完成！=====")
print(f"Eitlem格式数据根路径：{EITLEM_ROOT}")