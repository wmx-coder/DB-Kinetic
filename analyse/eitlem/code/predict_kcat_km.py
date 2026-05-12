import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from dataset2 import EitlemDataSet, EitlemDataLoader
from ensemble import ensemble  # 🔥 这里用的是 KKM 集成模型

# ===================== 配置 =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
protein_path = "/mnt/usb1/wmx/eitlem/Data/Feature/esm2_t33_650M_UR50D_unique"

# 你的小数据集（kcat/km 合并文件）
INPUT_PKL = "/mnt/usb1/wmx/eitlem/kcat/kcat-km_data_filtered_N20.pkl"

# KKM 模型根目录
KKM_MODEL_ROOT = "/mnt/usb1/wmx/eitlem/Results/KKM"

# 临时目录
TEMP_DIR = "/mnt/usb1/wmx/eitlem/kcat/analysis"
os.makedirs(TEMP_DIR, exist_ok=True)

# ===================== 转换数据为 KKM 格式 =====================
def convert_kkm_dataset(df):
    print("正在转换 KKM 数据集...")

    # 计算真实 kcat/Km（和你训练完全一致）
    df["kcat_over_Km"] = df["kcat(s^-1)"] / df["Km(M)"]
    label_col = "kcat_over_Km"

    # 底物编码（完全复刻你的预处理）
    df["sbt_feat_hash"] = df["sbt_feat"].apply(lambda x: hash(bytes(np.array(x, dtype=np.float32))))
    unique_hashes = list(set(df["sbt_feat_hash"]))
    hash2id = {h: i for i, h in enumerate(unique_hashes)}
    df["sbt_id"] = df["sbt_feat_hash"].map(hash2id)

    # 底物字典
    smiles_dict = {}
    for h in unique_hashes:
        row = df[df["sbt_feat_hash"] == h].iloc[0]
        vec = torch.FloatTensor(row["sbt_feat"][-167:])
        smiles_dict[hash2id[h]] = vec

    dict_path = os.path.join(TEMP_DIR, "KKM_dict.pt")
    torch.save(smiles_dict, dict_path)

    # pair_info: (esm_id, sbt_id, label)
    pair_info = []
    for _, row in df.iterrows():
        esm_id = str(row["esm_id"]).replace("kcat_", "").replace("km_", "")
        sbt_id = row["sbt_id"]
        label = float(row[label_col])
        pair_info.append((esm_id, sbt_id, label))

    return pair_info, df, dict_path

# ===================== 按 Fold 取 KKM 模型 =====================
def get_kkm_model_path(fold):
    fold_dir = os.path.join(KKM_MODEL_ROOT, f"Fold{fold}")
    train_dir = [d for d in os.listdir(fold_dir) if d.startswith("Transfer-")][0]
    weight_dir = os.path.join(fold_dir, train_dir, "Weight")
    pth = [f for f in os.listdir(weight_dir) if f.endswith(".pth")][0]
    return os.path.join(weight_dir, pth)

# ===================== KKM 预测 =====================
def predict_kkm(pair_info, df, smiles_dict_path):
    print("\n开始预测 KKM (kcat/Km)...")

    # 初始化 KKM 集成模型
    model = ensemble(167, 512, 1280, 10, 0.5, 10)
    df["KKM_pred"] = None

    for fold in tqdm(range(10), desc="Predict KKM"):
        indices = [i for i, _ in enumerate(pair_info) if df.iloc[i]["fold"] == fold]
        if not indices:
            continue

        sub_pair = [pair_info[i] for i in indices]
        model_path = get_kkm_model_path(fold)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        dataset = EitlemDataSet(sub_pair, protein_path, smiles_dict_path, log10=True)
        loader = EitlemDataLoader(dataset, batch_size=128, num_workers=0)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                preds.extend(out.cpu().numpy().flatten())

        # 从 log10 还原为真实 kcat/Km
        # preds = np.power(10, preds)
        df.loc[df["fold"] == fold, "KKM_pred"] = preds

    return df

# ===================== 主程序 =====================
if __name__ == "__main__":
    df = pd.read_pickle(INPUT_PKL)

    # 转换
    pair_info, df, dict_path = convert_kkm_dataset(df)

    # 预测
    df = predict_kkm(pair_info, df, dict_path)

    # 保存结果
    out_path = "/mnt/usb1/wmx/eitlem/kcat/N20_KKM_final_pred.pkl"
    df.to_pickle(out_path)

    print("\n🎉 KKM 预测完成！")
    print(f"输出文件：{out_path}")
    print("新增列：KKM_pred → 预测的 kcat/Km 真实值")