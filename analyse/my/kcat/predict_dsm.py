import torch as th
import pandas as pd
import numpy as np
import os
from argparse import RawDescriptionHelpFormatter
import argparse
from torch.utils.data import DataLoader, Dataset

# 导入自定义模块
from model_freeze import Model_Regression, ActivityModel_Freeze as ActivityModel
from util_check import RMSELoss

import logging
logging.basicConfig(level=logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ------------------------- 数据集类（无需fold） -------------------------
class KcatInferDataset(Dataset):
    def __init__(self, ezy_feats, sbt_feats):
        self.ezy_feats = ezy_feats
        self.sbt_feats = sbt_feats

    def __getitem__(self, idx):
        ezy = th.from_numpy(self.ezy_feats[idx]).float()
        sbt = th.from_numpy(self.sbt_feats[idx]).float()
        return ezy, sbt

    def __len__(self):
        return len(self.ezy_feats)

def collate_fn(batch):
    ezy_list, sbt_list = zip(*batch)
    batch_size = len(ezy_list)
    max_seq_len = max(ezy.shape[0] for ezy in ezy_list)

    ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
    enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
    for i, ezy in enumerate(ezy_list):
        seq_len = ezy.shape[0]
        ezy_padded[i, :seq_len, :] = ezy
        enzyme_mask[i, :, :seq_len] = 1

    sbt_tensor = th.stack(sbt_list, dim=0)
    return ezy_padded, sbt_tensor, enzyme_mask

# ------------------------- 加载 10 个折模型 -------------------------
def load_all_10_folds(model_save_dir, kcat_km_model_dir, km_model_dir, alpha, device):
    models = []
    for fold in range(10):
        print(f"Loading fold {fold} ...")
        kcat_km_ckpt = th.load(os.path.join(kcat_km_model_dir, f"fold{fold}_best_params.pth"), map_location=device)
        km_ckpt = th.load(os.path.join(km_model_dir, f"fold{fold}_best_params.pth"), map_location=device)
        model_ckpt = th.load(os.path.join(model_save_dir, f"fold{fold}_best_model.pth"), map_location=device)

        kcat_km_model = Model_Regression().to(device)
        kcat_km_model.load_state_dict(kcat_km_ckpt["model_state"])

        km_model = Model_Regression().to(device)
        km_model.load_state_dict(km_ckpt["model_state"])

        model = ActivityModel(
            kcat_km_model=kcat_km_model,
            Km_model=km_model,
            kcat_km_compress_state=kcat_km_ckpt["compress_state"],
            km_compress_state=km_ckpt["compress_state"],
            alpha=alpha,
            device=device
        ).to(device)
        model.load_state_dict(model_ckpt["model_state"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        models.append(model)
    return models

# ------------------------- 批量预测（10 折平均） -------------------------
def predict_batch(models, ezy_list, sbt_feat, batch_size, device):
    dataset = KcatInferDataset(ezy_list, sbt_feat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    preds = []

    with th.no_grad():
        for ezy, sbt, mask in dataloader:
            ezy = ezy.to(device)
            sbt = sbt.to(device)
            mask = mask.to(device)
            fold_preds = []
            for m in models:
                p, _, _ = m(ezy, sbt, mask)
                fold_preds.append(p.cpu().numpy().reshape(-1))
            avg = np.mean(fold_preds, axis=0)
            preds.append(avg)
    return np.concatenate(preds)

# ------------------------- 主函数：批量处理 6 个文件 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-alpha", type=float, default=0.5)   # 【修复】之前多余引号bug
    args = parser.parse_args()

    # ==================== 模型路径 ====================
    MODEL_SAVE_DIR    = "/mnt/usb1/wmx/catapro/models/kcat/models_model_freeze"
    KCAT_KM_DIR       = "/mnt/usb1/wmx/catapro/models/kcat_km/model3_1"
    KM_DIR            = "/mnt/usb1/wmx/catapro/models/km/km_model2_1"

    # ==================== 全部数据集 ====================
    FILES = [
        "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats.pkl"
    ]

    # 【单底物名单】只有这3个，只有 sbt_feat，不分1/2
    SINGLE_SUB_FILES = {
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl"
    }

    # 加载全部10折模型
    print("\nLoading all 10 folds ...")
    models = load_all_10_folds(MODEL_SAVE_DIR, KCAT_KM_DIR, KM_DIR, args.alpha, args.device)

    # 逐个预测
    for f in FILES:
        print("\n==================================================")
        print(f"Processing: {os.path.basename(f)}")

        df = pd.read_pickle(f)
        ezy_list = df["new_ezy_feat"].tolist()

        if f in SINGLE_SUB_FILES:
            # ========== 单底物：只用 sbt_feat，预测一次，三列结果完全相同 ==========
            sbt = np.vstack(df["sbt_feat"])
            pred = predict_batch(models, ezy_list, sbt, args.batch_size, args.device)
            p1 = p2 = final = pred
        else:
            # ========== 双底物：原有逻辑不变 ==========
            p1 = predict_batch(models, ezy_list, np.vstack(df["sbt_feat1"]), args.batch_size, args.device)
            p2 = predict_batch(models, ezy_list, np.vstack(df["sbt_feat2"]), args.batch_size, args.device)
            final = (p1 + p2) / 2

        # 输出列完全统一，后续代码不用动
        df["pred_log10_kcat_sub1"] = p1
        df["pred_log10_kcat_sub2"] = p2
        df["pred_log10_kcat_final"] = final

        out_path = f.replace(".pkl", "_pred_my.pkl")
        df.to_pickle(out_path)
        print(f"✅ Saved to: {out_path}")

    print("\n🎉 All 6 files finished!")