import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from dataset2 import EitlemDataSet, EitlemDataLoader
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor

# ===================== 配置 =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
protein_path = "/mnt/usb1/wmx/eitlem/Data/Feature/esm2_t33_650M_UR50D_unique"

KCAT_PKL = "/mnt/usb1/wmx/eitlem/kcat/kcat-data_filtered_N20.pkl"
KM_PKL = "/mnt/usb1/wmx/eitlem/kcat/km-data_filtered_N20.pkl"

KCAT_MODEL_ROOT = "/mnt/usb1/wmx/eitlem/Result/KCAT"
KM_MODEL_ROOT = "/mnt/usb1/wmx/eitlem/Result/KM"

TEMP_DIR = "/mnt/usb1/wmx/eitlem/_temp_pred"
os.makedirs(TEMP_DIR, exist_ok=True)

# ===================== 数据转换（完全对齐你训练代码） =====================
def convert_dataset(df, model_type):
    print(f"正在转换：{model_type}")

    if model_type == "KCAT":
        label_col = "kcat(s^-1)"
    else:
        label_col = "Km(M)"

    df["sbt_feat_hash"] = df["sbt_feat"].apply(lambda x: hash(bytes(np.array(x, dtype=np.float32))))
    unique_hashes = list(set(df["sbt_feat_hash"]))
    hash2id = {h: i for i, h in enumerate(unique_hashes)}
    df["sbt_id"] = df["sbt_feat_hash"].map(hash2id)

    smiles_dict = {}
    for h in unique_hashes:
        row = df[df["sbt_feat_hash"] == h].iloc[0]
        vec = torch.FloatTensor(row["sbt_feat"][-167:])
        smiles_dict[hash2id[h]] = vec
    torch.save(smiles_dict, os.path.join(TEMP_DIR, f"{model_type}_dict.pt"))

    pair_info = []
    for _, row in df.iterrows():
        esm_id = str(row["esm_id"]).replace("kcat_", "").replace("km_", "")
        sbt_id = row["sbt_id"]
        label = float(row[label_col])
        pair_info.append((esm_id, sbt_id, label))

    return pair_info, df

# ===================== 模型路径 =====================
def get_model_path(root, fold):
    fold_dir = os.path.join(root, f"Fold{fold}")
    train_dir = [d for d in os.listdir(fold_dir) if d.startswith("Transfer-")][0]
    weight_dir = os.path.join(fold_dir, train_dir, "Weight")
    pth = [f for f in os.listdir(weight_dir) if f.endswith(".pth")][0]
    return os.path.join(weight_dir, pth)

# ===================== 预测（完全对齐训练） =====================
def predict_task(model_type, pair_info, df):
    print(f"\n开始预测：{model_type}")

    if model_type == "KCAT":
        model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        model_root = KCAT_MODEL_ROOT
    else:
        model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
        model_root = KM_MODEL_ROOT

    smiles_dict = os.path.join(TEMP_DIR, f"{model_type}_dict.pt")
    df["pred"] = None

    for fold in tqdm(range(10), desc=f"Predict {model_type}"):
        indices = [i for i, p in enumerate(pair_info) if df.iloc[i]["fold"] == fold]
        if not indices: continue

        sub_pair = [pair_info[i] for i in indices]
        model_path = get_model_path(model_root, fold)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        dataset = EitlemDataSet(sub_pair, protein_path, smiles_dict, log10=True)
        loader = EitlemDataLoader(dataset, batch_size=128, num_workers=0)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)  # 🔥 关键修复1：整个batch丢进GPU
                out = model(batch)        # 🔥 关键修复2：只传batch！！！
                preds.extend(out.cpu().numpy().flatten())

        df.loc[df["fold"] == fold, "pred"] = preds
    return df

# ===================== 运行 =====================
if __name__ == "__main__":
    # KCAT
    kcat_df = pd.read_pickle(KCAT_PKL)
    kcat_pair, kcat_df = convert_dataset(kcat_df, "KCAT")
    kcat_out = predict_task("KCAT", kcat_pair, kcat_df)
    kcat_out.to_pickle("/mnt/usb1/wmx/eitlem/kcat/kcat_N20_final_pred.pkl")

    # KM
    km_df = pd.read_pickle(KM_PKL)
    km_pair, km_df = convert_dataset(km_df, "KM")
    km_out = predict_task("KM", km_pair, km_df)
    km_out.to_pickle("/mnt/usb1/wmx/eitlem/kcat/km_N20_final_pred.pkl")

    print("\n🎉 恭喜！预测全部成功完成！")