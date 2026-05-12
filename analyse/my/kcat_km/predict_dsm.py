# # import torch as th
# # import pandas as pd
# # import numpy as np
# # import os
# # from argparse import RawDescriptionHelpFormatter
# # import argparse
# # from torch.utils.data import DataLoader, Dataset
# #
# # # 导入自定义模块
# # from model_freeze import Model_Regression, ActivityModel_Freeze as ActivityModel
# # from util_check import RMSELoss
# #
# # import logging
# # logging.basicConfig(level=logging.ERROR)
# # from transformers import logging as hf_logging
# # hf_logging.set_verbosity_error()
# #
# # # ------------------------- 数据集类（无需fold） -------------------------
# # class KcatInferDataset(Dataset):
# #     def __init__(self, ezy_feats, sbt_feats):
# #         self.ezy_feats = ezy_feats
# #         self.sbt_feats = sbt_feats
# #
# #     def __getitem__(self, idx):
# #         ezy = th.from_numpy(self.ezy_feats[idx]).float()
# #         sbt = th.from_numpy(self.sbt_feats[idx]).float()
# #         return ezy, sbt
# #
# #     def __len__(self):
# #         return len(self.ezy_feats)
# #
# # def collate_fn(batch):
# #     ezy_list, sbt_list = zip(*batch)
# #     batch_size = len(ezy_list)
# #     max_seq_len = max(ezy.shape[0] for ezy in ezy_list)
# #
# #     ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
# #     enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
# #     for i, ezy in enumerate(ezy_list):
# #         seq_len = ezy.shape[0]
# #         ezy_padded[i, :seq_len, :] = ezy
# #         enzyme_mask[i, :, :seq_len] = 1
# #
# #     sbt_tensor = th.stack(sbt_list, dim=0)
# #     return ezy_padded, sbt_tensor, enzyme_mask
# #
# # # ------------------------- 加载 10 个折模型 -------------------------
# # def load_all_10_folds(model_save_dir, kcat_km_model_dir, km_model_dir, alpha, device):
# #     models = []
# #     for fold in range(10):
# #         print(f"Loading fold {fold} ...")
# #         kcat_km_ckpt = th.load(os.path.join(kcat_km_model_dir, f"fold{fold}_best_params.pth"), map_location=device)
# #         km_ckpt = th.load(os.path.join(km_model_dir, f"fold{fold}_best_params.pth"), map_location=device)
# #         model_ckpt = th.load(os.path.join(model_save_dir, f"fold{fold}_best_model.pth"), map_location=device)
# #
# #         kcat_km_model = Model_Regression().to(device)
# #         kcat_km_model.load_state_dict(kcat_km_ckpt["model_state"])
# #
# #         km_model = Model_Regression().to(device)
# #         km_model.load_state_dict(km_ckpt["model_state"])
# #
# #         model = ActivityModel(
# #             kcat_km_model=kcat_km_model,
# #             Km_model=km_model,
# #             kcat_km_compress_state=kcat_km_ckpt["compress_state"],
# #             km_compress_state=km_ckpt["compress_state"],
# #             alpha=alpha,
# #             device=device
# #         ).to(device)
# #         model.load_state_dict(model_ckpt["model_state"])
# #         model.eval()
# #         for p in model.parameters():
# #             p.requires_grad = False
# #         models.append(model)
# #     return models
# #
# # # ------------------------- 批量预测（10 折平均） -------------------------
# # def predict_batch(models, ezy_list, sbt_feat, batch_size, device):
# #     dataset = KcatInferDataset(ezy_list, sbt_feat)
# #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
# #     preds = []
# #
# #     with th.no_grad():
# #         for ezy, sbt, mask in dataloader:
# #             ezy = ezy.to(device)
# #             sbt = sbt.to(device)
# #             mask = mask.to(device)
# #             fold_preds = []
# #             for m in models:
# #                 p, _, _ = m(ezy, sbt, mask)
# #                 fold_preds.append(p.cpu().numpy().reshape(-1))
# #             avg = np.mean(fold_preds, axis=0)
# #             preds.append(avg)
# #     return np.concatenate(preds)
# #
# # # ------------------------- 主函数：批量处理 6 个文件 -------------------------
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
# #     parser.add_argument("-device", type=str, default="cuda:0")
# #     parser.add_argument("-batch_size", type=int, default=8)
# #     parser.add_argument("-alpha", type=float, default=0.5)   # 【修复】之前多余引号bug
# #     args = parser.parse_args()
# #
# #     # ==================== 模型路径 ====================
# #     MODEL_SAVE_DIR    = "/mnt/usb1/wmx/catapro/models/kcat/models_model_freeze"
# #     KCAT_KM_DIR       = "/mnt/usb1/wmx/catapro/models/kcat_km/model3_1"
# #     KM_DIR            = "/mnt/usb1/wmx/catapro/models/km/km_model2_1"
# #
# #     # ==================== 全部数据集 ====================
# #     FILES = [
# #         "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats.pkl"
# #     ]
# #
# #     # 【单底物名单】只有这3个，只有 sbt_feat，不分1/2
# #     SINGLE_SUB_FILES = {
# #         "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
# #         "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl"
# #     }
# #
# #     # 加载全部10折模型
# #     print("\nLoading all 10 folds ...")
# #     models = load_all_10_folds(MODEL_SAVE_DIR, KCAT_KM_DIR, KM_DIR, args.alpha, args.device)
# #
# #     # 逐个预测
# #     for f in FILES:
# #         print("\n==================================================")
# #         print(f"Processing: {os.path.basename(f)}")
# #
# #         df = pd.read_pickle(f)
# #         ezy_list = df["new_ezy_feat"].tolist()
# #
# #         if f in SINGLE_SUB_FILES:
# #             # ========== 单底物：只用 sbt_feat，预测一次，三列结果完全相同 ==========
# #             sbt = np.vstack(df["sbt_feat"])
# #             pred = predict_batch(models, ezy_list, sbt, args.batch_size, args.device)
# #             p1 = p2 = final = pred
# #         else:
# #             # ========== 双底物：原有逻辑不变 ==========
# #             p1 = predict_batch(models, ezy_list, np.vstack(df["sbt_feat1"]), args.batch_size, args.device)
# #             p2 = predict_batch(models, ezy_list, np.vstack(df["sbt_feat2"]), args.batch_size, args.device)
# #             final = (p1 + p2) / 2
# #
# #         # 输出列完全统一，后续代码不用动
# #         df["pred_log10_kcat_sub1"] = p1
# #         df["pred_log10_kcat_sub2"] = p2
# #         df["pred_log10_kcat_final"] = final
# #
# #         out_path = f.replace(".pkl", "_pred_my.pkl")
# #         df.to_pickle(out_path)
# #         print(f"✅ Saved to: {out_path}")
# #
# #     print("\n🎉 All 6 files finished!")
#
# import pandas as pd
# import numpy as np
# import torch as th
# import os
# from torch.utils.data import DataLoader, Dataset
# from argparse import RawDescriptionHelpFormatter
# import argparse
# import logging
#
# logging.basicConfig(level=logging.ERROR)
#
# from util_check import RMSELoss, rmse_loss, evaluate
# from model_kcat_km_1 import Model_Regression, KcatOriginalActivityModel, ActivityModel_Freeze as ActivityModel
#
#
# # ------------------------- 数据集类 -------------------------
# class Mydatasets_Infer(Dataset):
#     def __init__(self, ezy_feats, sbt_feats):
#         self.ezy_feats = ezy_feats
#         self.sbt_feats = sbt_feats
#
#     def __getitem__(self, idx):
#         ezy = th.from_numpy(self.ezy_feats[idx]).float()
#         sbt = th.from_numpy(self.sbt_feats[idx]).float()
#         return ezy, sbt
#
#     def __len__(self):
#         return len(self.ezy_feats)
#
#
# # ------------------------- 单底物 加载 -------------------------
# def load_infer_data_single(fpath):
#     full_df = pd.read_pickle(fpath).copy()
#     if "ezt" in full_df.columns and "ezy_feat" not in full_df.columns:
#         full_df.rename(columns={"ezt": "ezy_feat"}, inplace=True)
#
#     ezy_feats_list = []
#     invalid_ezy_idx = []
#     for idx, feat in enumerate(full_df["new_ezy_feat"]):
#         if isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024:
#             ezy_feats_list.append(feat)
#         else:
#             invalid_ezy_idx.append(idx)
#             ezy_feats_list.append(np.zeros((1, 1024)))
#
#     sbt_feats = []
#     invalid_sbt_idx = []
#     for idx, feat in enumerate(full_df["sbt_feat"]):
#         if isinstance(feat, np.ndarray) and len(feat) == 935:
#             sbt_feats.append(feat)
#         else:
#             invalid_sbt_idx.append(idx)
#             sbt_feats.append(np.zeros(935))
#     sbt_feats = np.array(sbt_feats)
#
#     invalid_idx = list(set(invalid_ezy_idx + invalid_sbt_idx))
#     return full_df, ezy_feats_list, sbt_feats, invalid_idx
#
#
# # ------------------------- 双底物 加载（sub1 + sub2） -------------------------
# def load_infer_data_double(fpath):
#     full_df = pd.read_pickle(fpath).copy()
#     if "ezt" in full_df.columns and "ezy_feat" not in full_df.columns:
#         full_df.rename(columns={"ezt": "ezy_feat"}, inplace=True)
#
#     ezy_feats_list = []
#     invalid_ezy_idx = []
#     for idx, feat in enumerate(full_df["new_ezy_feat"]):
#         if isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024:
#             ezy_feats_list.append(feat)
#         else:
#             invalid_ezy_idx.append(idx)
#             ezy_feats_list.append(np.zeros((1, 1024)))
#
#     # substrate 1
#     sbt1_feats = []
#     invalid_sbt1_idx = []
#     for idx, feat in enumerate(full_df["sbt_feat1"]):
#         if isinstance(feat, np.ndarray) and len(feat) == 935:
#             sbt1_feats.append(feat)
#         else:
#             invalid_sbt1_idx.append(idx)
#             sbt1_feats.append(np.zeros(935))
#     sbt1_feats = np.array(sbt1_feats)
#
#     # substrate 2
#     sbt2_feats = []
#     invalid_sbt2_idx = []
#     for idx, feat in enumerate(full_df["sbt_feat2"]):
#         if isinstance(feat, np.ndarray) and len(feat) == 935:
#             sbt2_feats.append(feat)
#         else:
#             invalid_sbt2_idx.append(idx)
#             sbt2_feats.append(np.zeros(935))
#     sbt2_feats = np.array(sbt2_feats)
#
#     invalid_idx = list(set(invalid_ezy_idx + invalid_sbt1_idx + invalid_sbt2_idx))
#     return full_df, ezy_feats_list, sbt1_feats, sbt2_feats, invalid_idx
#
#
# def collate_fn_infer(batch):
#     ezy_list, sbt_list = zip(*batch)
#     batch_size = len(ezy_list)
#     max_seq_len = max(ezy.shape[0] for ezy in ezy_list)
#
#     ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
#     enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
#     for i, ezy in enumerate(ezy_list):
#         seq_len = ezy.shape[0]
#         ezy_padded[i, :seq_len, :] = ezy
#         enzyme_mask[i, :, :seq_len] = 1
#
#     sbt_tensor = th.stack(sbt_list, dim=0)
#     return ezy_padded, sbt_tensor, enzyme_mask
#
#
# # ------------------------- 10 折预测 -------------------------
# def infer_10fold_fusion(
#         ezy_feats_list, sbt_feats, invalid_idx,
#         kcat_model_dir, km_model_dir, trained_model_dir,
#         device, batch_size=8, alpha=0.5, n_folds=10
# ):
#     n_samples = len(ezy_feats_list)
#     fold_fusion_preds = np.full((n_folds, n_samples), np.nan, dtype=np.float32)
#
#     for fold in range(n_folds):
#         print(f"→ Fold {fold}...", end="")
#         model_ckpt_path = os.path.join(trained_model_dir, f"fold{fold}_best_model.pth")
#         kcat_ckpt_path = os.path.join(kcat_model_dir, f"fold{fold}_best_params.pth")
#         km_ckpt_path = os.path.join(km_model_dir, f"fold{fold}_best_params.pth")
#
#         if not os.path.exists(model_ckpt_path) or not os.path.exists(kcat_ckpt_path) or not os.path.exists(
#                 km_ckpt_path):
#             print(" 跳过")
#             continue
#
#         kcat_ckpt = th.load(kcat_ckpt_path, map_location=device)
#         km_ckpt = th.load(km_ckpt_path, map_location=device)
#
#         dummy_kcat = Model_Regression().to(device)
#         dummy_km = Model_Regression().to(device)
#
#         kcat_model = KcatOriginalActivityModel(
#             kcat_km_model=dummy_kcat,
#             Km_model=dummy_km,
#             kcat_km_compress_state=kcat_ckpt["kcat_km_compress_state"],
#             km_compress_state=kcat_ckpt["km_compress_state"],
#             alpha=0.5,
#             device=device
#         ).to(device)
#         kcat_model.load_state_dict(kcat_ckpt["model_state"])
#
#         km_model = Model_Regression().to(device)
#         km_model.load_state_dict(km_ckpt["model_state"])
#
#         model = ActivityModel(
#             kcat_model=kcat_model,
#             km_model=km_model,
#             km_compress_state=km_ckpt["compress_state"],
#             alpha=alpha,
#             device=device
#         ).to(device)
#
#         trained_ckpt = th.load(model_ckpt_path, map_location=device)
#         model.load_state_dict(trained_ckpt["model_state"])
#         model.eval()
#
#         dataset = Mydatasets_Infer(ezy_feats_list, sbt_feats)
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer)
#
#         res = []
#         with th.no_grad():
#             for b in loader:
#                 e, s, m = [x.to(device) for x in b]
#                 p, _, _ = model(e, s, m)
#                 res.extend(p.squeeze(-1).cpu().numpy().tolist())
#
#         fold_fusion_preds[fold] = res
#         print(" 完成")
#
#     avg = np.nanmean(fold_fusion_preds, axis=0)
#     avg[invalid_idx] = np.nan
#     return avg, np.power(10, avg, where=~np.isnan(avg))
#
#
# # ------------------------- 保存结果（已修改：输出到指定目录） -------------------------
# def save_result(full_df, pred1_log10, pred1_val, pred2_log10, pred2_val, final_log10, final_val, input_fpath, out_dir):
#     df = full_df.copy()
#
#     # 添加三列：sub1, sub2, final
#     df["pred_fusion_log10[kcat/Km]_sub1"] = pred1_log10
#     df["pred_fusion_kcat/Km_sub1"] = pred1_val
#
#     if pred2_log10 is not None:
#         df["pred_fusion_log10[kcat/Km]_sub2"] = pred2_log10
#         df["pred_fusion_kcat/Km_sub2"] = pred2_val
#     else:
#         df["pred_fusion_log10[kcat/Km]_sub2"] = pred1_log10
#         df["pred_fusion_kcat/Km_sub2"] = pred1_val
#
#     df["pred_fusion_log10[kcat/Km]_final"] = final_log10
#     df["pred_fusion_kcat/Km_final"] = final_val
#
#     # 删掉特征列
#     drop_cols = [c for c in ["ezy_feat", "sbt_feat", "sbt_feat1", "sbt_feat2", "new_ezy_feat"] if c in df.columns]
#     df = df.drop(columns=drop_cols)
#
#     # 生成输出文件名
#     base_name = os.path.basename(input_fpath).replace(".pkl", "")
#     out_pkl = os.path.join(out_dir, f"{base_name}_pred.pkl")
#     out_csv = os.path.join(out_dir, f"{base_name}_catapro.csv")
#
#     # 保存
#     df.to_pickle(out_pkl)
#     df.to_csv(out_csv, index=False, encoding="utf-8")
#
#     print(f"✅ PKL 保存到：{out_pkl}")
#     print(f"✅ CSV 保存到：{out_csv}\n")
#
#
# # ------------------------- 主程序 -------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
#     parser.add_argument("-device", default="cuda:0")
#     parser.add_argument("-batch_size", type=int, default=8)
#     parser.add_argument("-alpha", type=float, default=0.5)
#     parser.add_argument("-n_folds", type=int, default=10)
#
#     # 模型路径
#     parser.add_argument("-kcat_model_dir", default="/mnt/usb1/wmx/catapro/models/kcat_km/model_kcat_2")
#     parser.add_argument("-km_model_dir", default="/mnt/usb1/wmx/catapro/models/kcat_km/model_km_2")
#     parser.add_argument("-trained_model_dir", default="/mnt/usb1/wmx/catapro/models/kcat_km/kcat_km/1_3")
#
#     # 输出目录（你可以自己改这里）
#     parser.add_argument("-out_dir", default="/mnt/usb1/wmx/catapro/analyse/dms/kcat_km")
#
#     args = parser.parse_args()
#
#     # 自动创建输出文件夹
#     os.makedirs(args.out_dir, exist_ok=True)
#
#     # ==================== 数据集 ====================
#     FILES = [
#         "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats.pkl"
#     ]
#
#     SINGLE_SUB_FILES = {
#         "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl"
#     }
#
#     # 批量处理
#     for fpath in FILES:
#         print("=" * 60)
#         print(f"处理：{os.path.basename(fpath)}")
#         print("=" * 60)
#
#         if fpath in SINGLE_SUB_FILES:
#             print("📦 单底物模式")
#             df, ezy, sbt, invalid = load_infer_data_single(fpath)
#             log10, val = infer_10fold_fusion(
#                 ezy, sbt, invalid,
#                 args.kcat_model_dir, args.km_model_dir, args.trained_model_dir,
#                 args.device, args.batch_size, args.alpha, args.n_folds
#             )
#             pred2_log10, pred2_val = None, None
#             final_log10 = log10
#             final_val = val
#
#         else:
#             print("📦 双底物模式")
#             df, ezy, sbt1, sbt2, invalid = load_infer_data_double(fpath)
#             log10_1, val_1 = infer_10fold_fusion(
#                 ezy, sbt1, invalid,
#                 args.kcat_model_dir, args.km_model_dir, args.trained_model_dir,
#                 args.device, args.batch_size, args.alpha, args.n_folds
#             )
#             log10_2, val_2 = infer_10fold_fusion(
#                 ezy, sbt2, invalid,
#                 args.kcat_model_dir, args.km_model_dir, args.trained_model_dir,
#                 args.device, args.batch_size, args.alpha, args.n_folds
#             )
#             final_log10 = (log10_1 + log10_2) / 2
#             final_val = (val_1 + val_2) / 2
#
#         # 保存到指定目录
#         save_result(df,
#                     log10_1 if not fpath in SINGLE_SUB_FILES else log10,
#                     val_1 if not fpath in SINGLE_SUB_FILES else val,
#                     log10_2 if not fpath in SINGLE_SUB_FILES else None,
#                     val_2 if not fpath in SINGLE_SUB_FILES else None,
#                     final_log10, final_val,
#                     fpath,
#                     args.out_dir)
#
#     print("🎉 所有文件处理完成！")


import pandas as pd
import numpy as np
import torch as th
import os
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
import logging

logging.basicConfig(level=logging.ERROR)

from util_check import RMSELoss, rmse_loss, evaluate
from model_kcat_km_1 import Model_Regression, KcatOriginalActivityModel, ActivityModel_Freeze as ActivityModel


# ------------------------- 数据集类 -------------------------
class Mydatasets_Infer(Dataset):
    def __init__(self, ezy_feats, sbt_feats):
        self.ezy_feats = ezy_feats
        self.sbt_feats = sbt_feats

    def __getitem__(self, idx):
        ezy = th.from_numpy(self.ezy_feats[idx]).float()
        sbt = th.from_numpy(self.sbt_feats[idx]).float()
        return ezy, sbt

    def __len__(self):
        return len(self.ezy_feats)


# ------------------------- 单底物 加载 -------------------------
def load_infer_data_single(fpath):
    full_df = pd.read_pickle(fpath).copy()
    if "ezt" in full_df.columns and "ezy_feat" not in full_df.columns:
        full_df.rename(columns={"ezt": "ezy_feat"}, inplace=True)

    ezy_feats_list = []
    invalid_ezy_idx = []
    for idx, feat in enumerate(full_df["new_ezy_feat"]):
        if isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024:
            ezy_feats_list.append(feat)
        else:
            invalid_ezy_idx.append(idx)
            ezy_feats_list.append(np.zeros((1, 1024)))

    sbt_feats = []
    invalid_sbt_idx = []
    for idx, feat in enumerate(full_df["sbt_feat"]):
        if isinstance(feat, np.ndarray) and len(feat) == 935:
            sbt_feats.append(feat)
        else:
            invalid_sbt_idx.append(idx)
            sbt_feats.append(np.zeros(935))
    sbt_feats = np.array(sbt_feats)

    invalid_idx = list(set(invalid_ezy_idx + invalid_sbt_idx))
    return full_df, ezy_feats_list, sbt_feats, invalid_idx


# ------------------------- 双底物 加载（sub1 + sub2） -------------------------
def load_infer_data_double(fpath):
    full_df = pd.read_pickle(fpath).copy()
    if "ezt" in full_df.columns and "ezy_feat" not in full_df.columns:
        full_df.rename(columns={"ezt": "ezy_feat"}, inplace=True)

    ezy_feats_list = []
    invalid_ezy_idx = []
    for idx, feat in enumerate(full_df["new_ezy_feat"]):
        if isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024:
            ezy_feats_list.append(feat)
        else:
            invalid_ezy_idx.append(idx)
            ezy_feats_list.append(np.zeros((1, 1024)))

    # substrate 1
    sbt1_feats = []
    invalid_sbt1_idx = []
    for idx, feat in enumerate(full_df["sbt_feat1"]):
        if isinstance(feat, np.ndarray) and len(feat) == 935:
            sbt1_feats.append(feat)
        else:
            invalid_sbt1_idx.append(idx)
            sbt1_feats.append(np.zeros(935))
    sbt1_feats = np.array(sbt1_feats)

    # substrate 2
    sbt2_feats = []
    invalid_sbt2_idx = []
    for idx, feat in enumerate(full_df["sbt_feat2"]):
        if isinstance(feat, np.ndarray) and len(feat) == 935:
            sbt2_feats.append(feat)
        else:
            invalid_sbt2_idx.append(idx)
            sbt2_feats.append(np.zeros(935))
    sbt2_feats = np.array(sbt2_feats)

    invalid_idx = list(set(invalid_ezy_idx + invalid_sbt1_idx + invalid_sbt2_idx))
    return full_df, ezy_feats_list, sbt1_feats, sbt2_feats, invalid_idx


def collate_fn_infer(batch):
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


# ------------------------- 10 折预测 -------------------------
def infer_10fold_fusion(
        ezy_feats_list, sbt_feats, invalid_idx,
        kcat_model_dir, km_model_dir, trained_model_dir,
        device, batch_size=8, alpha=0.5, n_folds=10
):
    n_samples = len(ezy_feats_list)
    fold_fusion_preds = np.full((n_folds, n_samples), np.nan, dtype=np.float32)

    for fold in range(n_folds):
        print(f"→ Fold {fold}...", end="")
        model_ckpt_path = os.path.join(trained_model_dir, f"fold{fold}_best_model.pth")
        kcat_ckpt_path = os.path.join(kcat_model_dir, f"fold{fold}_best_model.pth")
        km_ckpt_path = os.path.join(km_model_dir, f"fold{fold}_best_params.pth")

        if not os.path.exists(model_ckpt_path) or not os.path.exists(kcat_ckpt_path) or not os.path.exists(
                km_ckpt_path):
            print(" 跳过")
            continue

        kcat_ckpt = th.load(kcat_ckpt_path, map_location=device)
        km_ckpt = th.load(km_ckpt_path, map_location=device)

        dummy_kcat = Model_Regression().to(device)
        dummy_km = Model_Regression().to(device)

        kcat_model = KcatOriginalActivityModel(
            kcat_km_model=dummy_kcat,
            Km_model=dummy_km,
            kcat_km_compress_state=kcat_ckpt["kcat_km_compress_state"],
            km_compress_state=kcat_ckpt["km_compress_state"],
            alpha=0.5,
            device=device
        ).to(device)
        kcat_model.load_state_dict(kcat_ckpt["model_state"])

        km_model = Model_Regression().to(device)
        km_model.load_state_dict(km_ckpt["model_state"])

        model = ActivityModel(
            kcat_model=kcat_model,
            km_model=km_model,
            km_compress_state=km_ckpt["compress_state"],
            alpha=alpha,
            device=device
        ).to(device)

        trained_ckpt = th.load(model_ckpt_path, map_location=device)
        model.load_state_dict(trained_ckpt["model_state"])
        model.eval()

        dataset = Mydatasets_Infer(ezy_feats_list, sbt_feats)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer)

        res = []
        with th.no_grad():
            for b in loader:
                e, s, m = [x.to(device) for x in b]
                p, _, _ = model(e, s, m)
                res.extend(p.squeeze(-1).cpu().numpy().tolist())

        fold_fusion_preds[fold] = res
        print(" 完成")

    avg = np.nanmean(fold_fusion_preds, axis=0)
    avg[invalid_idx] = np.nan
    return avg, np.power(10, avg, where=~np.isnan(avg))


# ------------------------- 保存结果 -------------------------
def save_result(full_df, pred1_log10, pred1_val, pred2_log10, pred2_val, final_log10, final_val, input_fpath, out_dir):
    df = full_df.copy()

    df["pred_fusion_log10[kcat/Km]_sub1"] = pred1_log10
    df["pred_fusion_kcat/Km_sub1"] = pred1_val

    if pred2_log10 is not None:
        df["pred_fusion_log10[kcat/Km]_sub2"] = pred2_log10
        df["pred_fusion_kcat/Km_sub2"] = pred2_val
    else:
        df["pred_fusion_log10[kcat/Km]_sub2"] = pred1_log10
        df["pred_fusion_kcat/Km_sub2"] = pred1_val

    df["pred_fusion_log10[kcat/Km]_final"] = final_log10
    df["pred_fusion_kcat/Km_final"] = final_val

    drop_cols = [c for c in ["ezy_feat", "sbt_feat", "sbt_feat1", "sbt_feat2", "new_ezy_feat"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    base_name = os.path.basename(input_fpath).replace(".pkl", "")
    out_pkl = os.path.join(out_dir, f"{base_name}_pred.pkl")
    df.to_pickle(out_pkl)
    print(f"✅ 已保存：{out_pkl}")


# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-device", default="cuda:0")
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-alpha", type=float, default=0.5)
    parser.add_argument("-n_folds", type=int, default=10)

    parser.add_argument("-kcat_model_dir", default="/mnt/usb1/wmx/catapro/models/kcat_km/model_kcat_2")
    parser.add_argument("-km_model_dir", default="/mnt/usb1/wmx/catapro/models/kcat_km/model_km_2")
    parser.add_argument("-trained_model_dir", default="/mnt/usb1/wmx/catapro/models/kcat_km/kcat_km/1_3")
    parser.add_argument("-out_dir", default="/mnt/usb1/wmx/catapro/analyse/dms/kcat_km")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    FILES = [
        "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats.pkl"
    ]

    # ✅ 修复：[] 列表，不是 {} 集合！
    SINGLE_SUB_FILES = [
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl"
    ]

    for fpath in FILES:
        print("\n==================================================")
        print(f"处理：{os.path.basename(fpath)}")
        print("==================================================")

        if fpath in SINGLE_SUB_FILES:
            print("📦 单底物模式")
            df, ezy, sbt, invalid = load_infer_data_single(fpath)
            log10, val = infer_10fold_fusion(
                ezy, sbt, invalid,
                args.kcat_model_dir, args.km_model_dir, args.trained_model_dir,
                args.device, args.batch_size, args.alpha, args.n_folds
            )
            pred2_log10, pred2_val = None, None
            final_log10 = log10
            final_val = val

        else:
            print("📦 双底物模式")
            df, ezy, sbt1, sbt2, invalid = load_infer_data_double(fpath)
            log10_1, val_1 = infer_10fold_fusion(
                ezy, sbt1, invalid,
                args.kcat_model_dir, args.km_model_dir, args.trained_model_dir,
                args.device, args.batch_size, args.alpha, args.n_folds
            )
            log10_2, val_2 = infer_10fold_fusion(
                ezy, sbt2, invalid,
                args.kcat_model_dir, args.km_model_dir, args.trained_model_dir,
                args.device, args.batch_size, args.alpha, args.n_folds
            )
            final_log10 = (log10_1 + log10_2) / 2
            final_val = (val_1 + val_2) / 2

        save_result(df,
                    log10_1 if not fpath in SINGLE_SUB_FILES else log10,
                    val_1 if not fpath in SINGLE_SUB_FILES else val,
                    log10_2 if not fpath in SINGLE_SUB_FILES else None,
                    val_2 if not fpath in SINGLE_SUB_FILES else None,
                    final_log10, final_val,
                    fpath,
                    args.out_dir)

    print("\n🎉 全部预测完成！")