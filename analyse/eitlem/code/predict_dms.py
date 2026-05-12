# import os
# import pandas as pd
# import torch
# from tqdm import tqdm
# import numpy as np
#
# # ===================== 【路径配置】你要的6个文件 =====================
# INPUT_FILES = {
#     "EcTL": "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats.pkl",
#     "HIS3": "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
#     "HIS7": "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
#     "si-dbs_ub": "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
#     "Ssdata_ub": "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats.pkl",
#     "Ttdata": "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats.pkl"
# }
#
# # 输出根目录 → 你指定的路径
# OUTPUT_ROOT = "/mnt/usb1/wmx/eitlem/dms"
# os.makedirs(OUTPUT_ROOT, exist_ok=True)
#
# # ESM 输出目录
# ESM_DIR = os.path.join(OUTPUT_ROOT, "esm2_t33_650M_UR50D")
# os.makedirs(ESM_DIR, exist_ok=True)
#
# # FASTA 路径
# FASTA_PATH = os.path.join(OUTPUT_ROOT, "seq_str.fasta")
#
# # ======================================================
# # 🔥 步骤1：加载6个数据集，提取唯一序列
# # ======================================================
# def load_all_sequences():
#     all_seqs = {}
#     datasets = {}
#
#     for name, path in INPUT_FILES.items():
#         df = pd.read_pickle(path)
#         datasets[name] = df
#
#         # 自动兼容大写 Sequence / 小写 sequence
#         seq_col = "Sequence" if "Sequence" in df.columns else "sequence"
#         # 去重
#         for seq in df[seq_col].drop_duplicates():
#             if seq not in all_seqs:
#                 all_seqs[seq] = len(all_seqs)
#
#     print(f"✅ 总唯一序列数：{len(all_seqs)}")
#     return all_seqs, datasets
#
# # ======================================================
# # 🔥 步骤2：生成 FASTA 文件
# # ======================================================
# def write_fasta(seq_dict):
#     with open(FASTA_PATH, "w") as f:
#         for seq, idx in seq_dict.items():
#             f.write(f">{idx}\n{seq}\n")
#     print(f"✅ FASTA 已生成：{FASTA_PATH}")
#
# # ======================================================
# # 🔥 步骤3：调用 extract.py 提取 ESM 特征
# # ======================================================
# def run_esm_extract():
#     cmd = f'python ./extract.py esm2_t33_650M_UR50D {FASTA_PATH} {ESM_DIR}/ --repr_layers 33 --include per_tok'
#     print("🚀 运行 ESM 特征提取：", cmd)
#     os.system(cmd)
#
# # ======================================================
# # 🔥 步骤4：处理特征，只保留 layer33
# # ======================================================
# def process_esm_features():
#     file_list = sorted([f for f in os.listdir(ESM_DIR) if f.endswith(".pt")],
#                        key=lambda x: int(x.replace(".pt", "")))
#
#     for fname in tqdm(file_list, desc="处理 ESM 特征"):
#         path = os.path.join(ESM_DIR, fname)
#         data = torch.load(path)
#         data = data['representations'][33]
#         torch.save(data, path)
#
#     print(f"✅ 所有 ESM 特征处理完成：{ESM_DIR}")
#
# # ======================================================
# # 🔥 步骤5：给每个数据集添加 esm_id 并保存
# # ======================================================
# def add_esm_id_and_save(seq_dict, datasets):
#     for name, df in datasets.items():
#         seq_col = "Sequence" if "Sequence" in df.columns else "sequence"
#         df["esm_id"] = df[seq_col].map(seq_dict)
#
#         out_path = os.path.join(OUTPUT_ROOT, f"{name}_with_esmid.pkl")
#         df.to_pickle(out_path)
#         print(f"✅ {name} 已添加 esm_id → {out_path}")
#
# # ======================================================
# # 🔥 主程序
# # ======================================================
# if __name__ == "__main__":
#     print("=" * 60)
#     print(" 开始处理 6 个 DMS 数据集 → 提取ESM特征 + 生成esm_id")
#     print("=" * 60)
#
#     # 1. 加载唯一序列
#     seq_dict, datasets = load_all_sequences()
#
#     # 2. 生成FASTA
#     write_fasta(seq_dict)
#
#     # 3. 提取ESM
#     run_esm_extract()
#
#     # 4. 处理ESM
#     process_esm_features()
#
#     # 5. 添加esm_id
#     add_esm_id_and_save(seq_dict, datasets)
#
#     print("\n🎉 全部完成！")
#     print(f"📁 ESM 特征：{ESM_DIR}")
#     print(f"📁 带 esm_id 数据集：{OUTPUT_ROOT}")

# import pandas as pd
# import os
#
# # ===================== 【配置】直接用你已有的文件 =====================
# OLD_ESMID_FILES = {
#     "HIS3":      "/mnt/usb1/wmx/eitlem/dms/HIS3_with_esmid.pkl",
#     "HIS7":      "/mnt/usb1/wmx/eitlem/dms/HIS7_with_esmid.pkl",
#     "si-dbs_ub": "/mnt/usb1/wmx/eitlem/dms/si-dbs_ub_with_esmid.pkl",
# }
#
# # 你现在的新文件（有 sbt_feat）
# NEW_FEAT_FILES = {
#     "HIS3":      "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
#     "HIS7":      "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
#     "si-dbs_ub": "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
# }
#
# # 输出最终文件（给 eitlem 预测用）
# OUTPUT_DIR = "/mnt/usb1/wmx/eitlem/dms/"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # ===================== 【核心】复制 esm_id =====================
# for name in OLD_ESMID_FILES:
#     print(f"\n===== 处理 {name} =====")
#
#     # 1. 读取旧文件（有 esm_id）
#     df_old = pd.read_pickle(OLD_ESMID_FILES[name])
#     # 2. 读取新文件（有 sbt_feat）
#     df_new = pd.read_pickle(NEW_FEAT_FILES[name])
#
#     # 3. 取出 序列 -> esm_id 映射
#     seq_col_old = "Sequence" if "Sequence" in df_old.columns else "sequence"
#     seq_col_new = "Sequence" if "Sequence" in df_new.columns else "sequence"
#
#     seq_to_esmid = dict(zip(df_old[seq_col_old], df_old["esm_id"]))
#
#     # 4. 直接给新文件加上 esm_id
#     df_new["esm_id"] = df_new[seq_col_new].map(seq_to_esmid)
#
#     # 5. 保存（覆盖原来的 with_esmid.pkl）
#     out_path = os.path.join(OUTPUT_DIR, f"{name}_with_esmid.pkl")
#     df_new.to_pickle(out_path)
#
#     print(f"✅ 已添加 esm_id → {out_path}")
#     print(f"   包含列：sbt_feat ✅ + esm_id ✅")
#
# print("\n🎉🎉🎉 全部完成！不用跑 ESM！直接去预测！")

# import torch
# import pandas as pd
# import os
# import numpy as np
# from dataset2 import EitlemDataSet, EitlemDataLoader
# from KCM import EitlemKcatPredictor
#
# # ===================== 路径配置 =====================
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# PROTEIN_FEAT_DIR = "/mnt/usb1/wmx/eitlem/dms/esm2_t33_650M_UR50D"
# SAVE_ROOT = "/mnt/usb1/wmx/catapro/analyse/dms"
# KCAT_MODEL_ROOT = "/mnt/usb1/wmx/eitlem/Result/KCAT"
#
# DATASET_PATHS = {
#     "EcTL": "/mnt/usb1/wmx/eitlem/dms/EcTL_with_esmid.pkl",
#     "HIS3": "/mnt/usb1/wmx/eitlem/dms/HIS3_with_esmid.pkl",
#     "HIS7": "/mnt/usb1/wmx/eitlem/dms/HIS7_with_esmid.pkl",
#     "si-dbs_ub": "/mnt/usb1/wmx/eitlem/dms/si-dbs_ub_with_esmid.pkl",
#     "Ssdata_ub": "/mnt/usb1/wmx/eitlem/dms/Ssdata_ub_with_esmid.pkl",
#     "Ttdata": "/mnt/usb1/wmx/eitlem/dms/Ttdata_with_esmid.pkl"
# }
#
# # 单底物文件（只有 sbt_feat，不分 1/2）
# SINGLE_SUBSTRATE = {"HIS3", "HIS7", "si-dbs_ub"}
#
# os.makedirs(SAVE_ROOT, exist_ok=True)
#
# # ===================== 底物处理 =====================
# def process_substrate(df, save_name, sbt_col="sbt_feat"):
#     df["sbt_feat_hash"] = df[sbt_col].apply(lambda x: hash(bytes(np.array(x, dtype=np.float32))))
#     unique_hashes = sorted(list(set(df["sbt_feat_hash"])))
#     hash2id = {h: i for i, h in enumerate(unique_hashes)}
#     df["sbt_id"] = df["sbt_feat_hash"].map(hash2id)
#
#     smiles_dict = {}
#     for h in unique_hashes:
#         row = df[df["sbt_feat_hash"] == h].iloc[0]
#         vec = torch.FloatTensor(row[sbt_col][-167:])
#         smiles_dict[hash2id[h]] = vec
#
#     dict_path = os.path.join(SAVE_ROOT, f"{save_name}_{sbt_col}_smiles_dict.pt")
#     torch.save(smiles_dict, dict_path)
#     return df, dict_path
#
# # ===================== 构建预测 pair =====================
# def build_pair_info(df):
#     pair_info = []
#     for _, row in df.iterrows():
#         esm_id = str(row["esm_id"])
#         sbt_id = row["sbt_id"]
#         pair_info.append((esm_id, sbt_id, 1.0))
#     return pair_info
#
# # ===================== 加载 10 个 Kcat 模型 =====================
# def load_all_kcat_models():
#     models = []
#     for fold in range(10):
#         try:
#             fold_dir = os.path.join(KCAT_MODEL_ROOT, f"Fold{fold}")
#             train_dir = [d for d in os.listdir(fold_dir) if d.startswith("Transfer-")][0]
#             weight_dir = os.path.join(fold_dir, train_dir, "Weight")
#             pth = [f for f in os.listdir(weight_dir) if f.endswith(".pth")][0]
#             model_path = os.path.join(weight_dir, pth)
#
#             model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
#             model.load_state_dict(torch.load(model_path, map_location=device))
#             model.to(device).eval()
#             models.append(model)
#             print(f"✅ Fold{fold} Kcat 模型加载成功")
#         except Exception as e:
#             print(f"⚠️ Fold{fold} 加载失败: {e}")
#     return models
#
# # ===================== 10 折平均预测 =====================
# def predict_10fold_mean(models, pair_info, smiles_dict_path):
#     all_preds = []
#
#     dataset = EitlemDataSet(pair_info, PROTEIN_FEAT_DIR, smiles_dict_path, log10=True)
#     loader = EitlemDataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
#
#     with torch.no_grad():
#         for model in models:
#             preds = []
#             for batch in loader:
#                 batch = batch.to(device)
#                 out = model(batch)
#                 preds.extend(out.cpu().numpy().flatten())
#             all_preds.append(preds)
#
#     mean_log = np.mean(np.array(all_preds), axis=0)
#     return mean_log
#
# # ===================== 预测主逻辑（自动单/双底物） =====================
# def predict_dataset(name, filepath, models):
#     print(f"\n===== 处理 {name} (Kcat 预测) =====")
#     df = pd.read_pickle(filepath)
#
#     if name in SINGLE_SUBSTRATE:
#         # ===================== 单底物：只预测一次 =====================
#         print("   → 单底物模式，使用 sbt_feat")
#         df1, dict1 = process_substrate(df, name, sbt_col="sbt_feat")
#         pair1 = build_pair_info(df1)
#         pred = predict_10fold_mean(models, pair1, dict1)
#
#         pred1 = pred
#         pred2 = pred
#         pred_final = pred
#
#     else:
#         # ===================== 双底物：正常预测 =====================
#         print("   → 双底物模式，使用 sbt_feat1 / sbt_feat2")
#         df1, dict1 = process_substrate(df.copy(), name, sbt_col="sbt_feat1")
#         pair1 = build_pair_info(df1)
#         pred1 = predict_10fold_mean(models, pair1, dict1)
#
#         df2, dict2 = process_substrate(df.copy(), name, sbt_col="sbt_feat2")
#         pair2 = build_pair_info(df2)
#         pred2 = predict_10fold_mean(models, pair2, dict2)
#         pred_final = (pred1 + pred2) / 2
#
#     # 保存结果（输出格式完全统一）
#     df["pred_kcat_sub1"] = pred1
#     df["pred_kcat_sub2"] = pred2
#     df["pred_kcat_final"] = pred_final
#
#     out_path = os.path.join(SAVE_ROOT, f"{name}_eitlem_kcat.pkl")
#     df.to_pickle(out_path)
#     print(f"✅ Kcat 预测完成 → {out_path}")
#     return df
#
# # ===================== 主程序 =====================
# if __name__ == "__main__":
#     print("🚀 加载所有 10 折 Kcat 模型...")
#     models = load_all_kcat_models()
#
#     for name, fp in DATASET_PATHS.items():
#         predict_dataset(name, fp, models)
#
#     print("\n🎉🎉🎉 6 个 DMS 数据集 Kcat 预测全部完成！")


import torch
import pandas as pd
import os
import numpy as np
from dataset2 import EitlemDataSet, EitlemDataLoader
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor

# ===================== 路径配置（完全对齐你的） =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PROTEIN_FEAT_DIR = "/mnt/usb1/wmx/eitlem/dms/esm2_t33_650M_UR50D"
SAVE_ROOT = "/mnt/usb1/wmx/catapro/analyse/dms"

KCAT_MODEL_ROOT = "/mnt/usb1/wmx/eitlem/Result/KCAT"
KM_MODEL_ROOT  = "/mnt/usb1/wmx/eitlem/Result/KM"

DATASET_PATHS = {
    "EcTL": "/mnt/usb1/wmx/eitlem/dms/EcTL_with_esmid.pkl",
    "HIS3": "/mnt/usb1/wmx/eitlem/dms/HIS3_with_esmid.pkl",
    "HIS7": "/mnt/usb1/wmx/eitlem/dms/HIS7_with_esmid.pkl",
    "si-dbs_ub": "/mnt/usb1/wmx/eitlem/dms/si-dbs_ub_with_esmid.pkl",
    "Ssdata_ub": "/mnt/usb1/wmx/eitlem/dms/Ssdata_ub_with_esmid.pkl",
    "Ttdata": "/mnt/usb1/wmx/eitlem/dms/Ttdata_with_esmid.pkl"
}

SINGLE_SUBSTRATE = {"HIS3", "HIS7", "si-dbs_ub"}
os.makedirs(SAVE_ROOT, exist_ok=True)

# ===================== 底物处理（和你能跑通的代码完全一样） =====================
def process_substrate(df, save_name, sbt_col="sbt_feat"):
    df["sbt_feat_hash"] = df[sbt_col].apply(lambda x: hash(bytes(np.array(x, dtype=np.float32))))
    unique_hashes = sorted(list(set(df["sbt_feat_hash"])))
    hash2id = {h: i for i, h in enumerate(unique_hashes)}
    df["sbt_id"] = df["sbt_feat_hash"].map(hash2id)

    smiles_dict = {}
    for h in unique_hashes:
        row = df[df["sbt_feat_hash"] == h].iloc[0]
        vec = torch.FloatTensor(row[sbt_col][-167:])
        smiles_dict[hash2id[h]] = vec

    dict_path = os.path.join(SAVE_ROOT, f"{save_name}_{sbt_col}_dict.pt")
    torch.save(smiles_dict, dict_path)
    return df, dict_path

def build_pair_info(df):
    pair_info = []
    for _, row in df.iterrows():
        esm_id = str(row["esm_id"])
        sbt_id = row["sbt_id"]
        pair_info.append((esm_id, sbt_id, 1.0))
    return pair_info

# ===================== 加载 10 个模型（你要的方式） =====================
def load_models(model_cls, model_root):
    models = []
    for fold in range(10):
        try:
            fold_dir = os.path.join(model_root, f"Fold{fold}")
            train_dir = [d for d in os.listdir(fold_dir) if d.startswith("Transfer-")][0]
            weight_dir = os.path.join(fold_dir, train_dir, "Weight")
            pth = [f for f in os.listdir(weight_dir) if f.endswith(".pth")][0]
            model_path = os.path.join(weight_dir, pth)

            model = model_cls(167, 512, 1280, 10, 0.5, 10)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device).eval()
            models.append(model)
            print(f"✅ Fold{fold} 加载成功")
        except Exception as e:
            print(f"❌ Fold{fold} 加载失败")
    return models

# ===================== 10 折直接平均预测（完全照搬你能跑的！） =====================
def predict_10fold_mean(models, pair_info, smiles_dict_path):
    all_preds = []

    dataset = EitlemDataSet(pair_info, PROTEIN_FEAT_DIR, smiles_dict_path, log10=True)
    loader = EitlemDataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)

    with torch.no_grad():
        for model in models:
            preds = []
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                preds.extend(out.cpu().numpy().flatten())
            all_preds.append(preds)

    mean_log = np.mean(np.array(all_preds), axis=0)
    return mean_log

# ===================== 预测一个数据集 =====================
def predict_one(name, filepath, kcat_models, km_models):
    print(f"\n===== 预测 {name} =====")
    df = pd.read_pickle(filepath)

    if name in SINGLE_SUBSTRATE:
        df1, d1 = process_substrate(df, name, "sbt_feat")
        pair = build_pair_info(df1)

        kcat = predict_10fold_mean(kcat_models, pair, d1)
        km   = predict_10fold_mean(km_models, pair, d1)

        k1, k2 = kcat, kcat
        m1, m2 = km, km
        kf = kcat
        mf = km

    else:
        # 底物1
        df1, d1 = process_substrate(df.copy(), name, "sbt_feat1")
        p1 = build_pair_info(df1)
        k1 = predict_10fold_mean(kcat_models, p1, d1)
        m1 = predict_10fold_mean(km_models, p1, d1)

        # 底物2
        df2, d2 = process_substrate(df.copy(), name, "sbt_feat2")
        p2 = build_pair_info(df2)
        k2 = predict_10fold_mean(kcat_models, p2, d2)
        m2 = predict_10fold_mean(km_models, p2, d2)

        kf = (k1 + k2) / 2
        mf = (m1 + m2) / 2

    # 保存结果
    df["pred_kcat_sub1"] = k1
    df["pred_kcat_sub2"] = k2
    df["pred_kcat_final"] = kf

    df["pred_km_sub1"] = m1
    df["pred_km_sub2"] = m2
    df["pred_km_final"] = mf

    out = os.path.join(SAVE_ROOT, f"{name}_Kcat_Km.pkl")
    df.to_pickle(out)
    print(f"✅ 保存完成：{out}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("🚀 加载 Kcat 模型...")
    kcat_ms = load_models(EitlemKcatPredictor, KCAT_MODEL_ROOT)

    print("\n🚀 加载 Km 模型...")
    km_ms = load_models(EitlemKmPredictor, KM_MODEL_ROOT)

    # 批量预测
    for name, fp in DATASET_PATHS.items():
        predict_one(name, fp, kcat_ms, km_ms)

    print("\n🎉🎉🎉 所有预测完成！")