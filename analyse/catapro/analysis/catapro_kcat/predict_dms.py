# import torch as th
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# from utils import *
# from model import *
# from act_model import KcatModel as _KcatModel
# import argparse
# import logging
# from transformers import logging as hf_logging
#
# hf_logging.set_verbosity_error()
# logging.basicConfig(level=logging.ERROR)
#
# # def inference_single_data(model, ezy_feats, sbt_feats, device="cuda:0"):
# #     model.eval()
# #     with th.no_grad():
# #         pred = model(ezy_feats, sbt_feats)
# #         if isinstance(pred, tuple):
# #             pred = pred[-1]
# #         pred = pred.cpu().numpy()
# #     return pred
#
# def inference_single_data(model, ezy_feats, sbt_feats, device="cuda:0"):
#     model.eval()
#     with th.no_grad():
#         pred = model(ezy_feats, sbt_feats)  # ✅ 直接输出值
#         pred = pred.cpu().numpy()
#     return pred
#
# if __name__ == "__main__":
#     # ===================== 固定配置 =====================
#     MODEL_PATH   = "/mnt/usb1/wmx/catapro/models/catapro/models"
#     DEVICE       = "cuda"
#     INPUT_DIR    = "/mnt/usb1/wmx/catapro/analyse/dms"
#     OUTPUT_DIR   = "/mnt/usb1/wmx/catapro/analyse/dms"
#
#     # ===================== 数据集 =====================
#     DATASETS = [
#         "EcTL_with_sub_feats.pkl",
#         "HIS3_feats.pkl",
#         "HIS7_feats.pkl",
#         "si-dbs_ub_feats.pkl",
#         "Ssdata_ub_feats.pkl",
#         "Ttdata_with_sub_feats.pkl"
#     ]
#
#     # 单底物文件（只有 sbt_feat，不分1/2）
#     SINGLE_SUBSTRATE_FILES = {
#         "HIS3_feats.pkl",
#         "HIS7_feats.pkl",
#         "si-dbs_ub_feats.pkl"
#     }
#
#     # ===================== 加载 10 折模型 =====================
#     print("Loading 10-fold models...")
#     model_list = []
#     kcat_path = f"{MODEL_PATH}/kcat_models"
#     for fold in range(10):
#         model = _KcatModel(device=DEVICE)
#         model.load_state_dict(th.load(f"{kcat_path}/{fold}_bestmodel.pth", map_location=DEVICE))
#         model_list.append(model)
#         print(f"✅ Fold {fold} loaded")
#
#     # ===================== 批量处理 =====================
#     for fname in DATASETS:
#         inp_path  = f"{INPUT_DIR}/{fname}"
#         out_path  = f"{OUTPUT_DIR}/{fname.replace('.pkl', '_pred.pkl')}"
#         # csv_path  = out_path.replace(".pkl", ".csv")
#
#         print("\n" + "="*60)
#         print(f"Processing: {fname}")
#         print(f"Input:  {inp_path}")
#         print(f"Output: {out_path}")
#         print("="*60)
#
#         df = pd.read_pickle(inp_path)
#         is_single = fname in SINGLE_SUBSTRATE_FILES
#
#         pred_sub1 = []
#         pred_sub2 = []
#         pred_final = []
#
#         total = len(df)
#         for i, (_, row) in enumerate(df.iterrows(), 1):
#             try:
#                 ezy = th.from_numpy(row["ezy_feat"]).float().to(DEVICE).unsqueeze(0)
#
#                 # ===================== 核心修改 =====================
#                 if is_single:
#                     # 单底物：只用 sbt_feat，sub1=sub2=最终值
#                     sbt = th.from_numpy(row["sbt_feat"]).float().to(DEVICE).unsqueeze(0)
#                     p = np.mean([inference_single_data(m, ezy, sbt, DEVICE)[0][0] for m in model_list])
#                     p1 = p
#                     p2 = p
#                     final = p
#                 else:
#                     # 双底物：原来逻辑不变
#                     s1 = th.from_numpy(row["sbt_feat1"]).float().to(DEVICE).unsqueeze(0)
#                     s2 = th.from_numpy(row["sbt_feat2"]).float().to(DEVICE).unsqueeze(0)
#                     p1 = np.mean([inference_single_data(m, ezy, s1, DEVICE)[0][0] for m in model_list])
#                     p2 = np.mean([inference_single_data(m, ezy, s2, DEVICE)[0][0] for m in model_list])
#                     final = (p1 + p2) / 2
#
#                 pred_sub1.append(p1)
#                 pred_sub2.append(p2)
#                 pred_final.append(final)
#
#                 if i % 200 == 0:
#                     print(f"Progress: {i}/{total}")
#
#             except Exception as e:
#                 pred_sub1.append(np.nan)
#                 pred_sub2.append(np.nan)
#                 pred_final.append(np.nan)
#
#         # 保存结果
#         df["pred_log10[kcat]_sub1"]  = pred_sub1
#         df["pred_log10[kcat]_sub2"]  = pred_sub2
#         df["pred_log10[kcat]_final"] = pred_final
#
#         df.to_pickle(out_path)
#
#         print(f"✅ Done: {fname}")
#
#     print("\n🎉 All files finished!")

import torch as th
import pandas as pd
import numpy as np
from utils import *
from model import *
from act_model import KcatModel as _KcatModel
from act_model import KmModel as _KmModel
from act_model import ActivityModel as _ActivityModel
import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)


# ===================== 【完全正确】推理函数 =====================
def inference(kcat_model, Km_model, act_model, ezy_feats, sbt_feats, device="cuda:0"):
    kcat_model.eval()
    Km_model.eval()
    act_model.eval()
    with th.no_grad():
        pred_kcat = kcat_model(ezy_feats, sbt_feats)[0].cpu().numpy()
        pred_Km = Km_model(ezy_feats, sbt_feats)[0].cpu().numpy()
        pred_act = act_model(ezy_feats, sbt_feats)[-1].cpu().numpy()
        return np.concatenate([pred_kcat, pred_Km, pred_act], axis=1)


# ===================== 主程序 =====================
if __name__ == "__main__":
    MODEL_PATH = "/mnt/usb1/wmx/catapro/models/catapro/models"
    DEVICE = "cuda"
    INPUT_DIR = "/mnt/usb1/wmx/catapro/analyse/dms"
    OUTPUT_DIR = "/mnt/usb1/wmx/catapro/analyse/dms"

    DATASETS = [
        "EcTL_with_sub_feats.pkl",
        "HIS3_feats.pkl",
        "HIS7_feats.pkl",
        "si-dbs_ub_feats.pkl",
        "Ssdata_ub_feats.pkl",
        "Ttdata_with_sub_feats.pkl"
    ]

    SINGLE_SUBSTRATE_FILES = {
        "HIS3_feats.pkl",
        "HIS7_feats.pkl",
        "si-dbs_ub_feats.pkl"
    }

    # ===================== 加载 10 折模型 =====================
    print("Loading 10-fold models...")
    kcat_list = []
    km_list = []
    act_list = []

    for fold in range(10):
        m1 = _KcatModel(device=DEVICE)
        m1.load_state_dict(th.load(f"{MODEL_PATH}/kcat_models/{fold}_bestmodel.pth", map_location=DEVICE))
        kcat_list.append(m1)

        m2 = _KmModel(device=DEVICE)
        m2.load_state_dict(th.load(f"{MODEL_PATH}/Km_models/{fold}_bestmodel.pth", map_location=DEVICE))
        km_list.append(m2)

        m3 = _ActivityModel(device=DEVICE)
        m3.load_state_dict(th.load(f"{MODEL_PATH}/act_models/{fold}_bestmodel.pth", map_location=DEVICE))
        act_list.append(m3)

        print(f"✅ Fold {fold} loaded")

    # ===================== 批量预测 =====================
    for fname in DATASETS:
        inp_path = f"{INPUT_DIR}/{fname}"
        out_path = f"{OUTPUT_DIR}/{fname.replace('.pkl', '_pred.pkl')}"

        print(f"\nProcessing: {fname}")
        df = pd.read_pickle(inp_path)
        is_single = fname in SINGLE_SUBSTRATE_FILES

        ezy_feat = np.stack(df["ezy_feat"].values)
        ezy_feat = th.tensor(ezy_feat, dtype=th.float32).to(DEVICE)

        if is_single:
            sbt_feat = np.stack(df["sbt_feat"].values)
            sbt_feat = th.tensor(sbt_feat, dtype=th.float32).to(DEVICE)

            p1_all, km_all, a_all = [], [], []
            for kc, km, act in zip(kcat_list, km_list, act_list):
                pred = inference(kc, km, act, ezy_feat, sbt_feat, DEVICE)
                p1_all.append(pred[:, [0]])
                km_all.append(pred[:, [1]])
                a_all.append(pred[:, [2]])

            p1 = np.mean(np.concatenate(p1_all, axis=1), axis=1)
            km_val = np.mean(np.concatenate(km_all, axis=1), axis=1)
            a = np.mean(np.concatenate(a_all, axis=1), axis=1)

            df["pred_log10[kcat]_sub1"] = p1
            df["pred_log10[kcat]_sub2"] = p1
            df["pred_log10[Km]_sub1"] = km_val
            df["pred_log10[Km]_sub2"] = km_val
            df["pred_log10[kcat/Km]_sub1"] = a
            df["pred_log10[kcat/Km]_sub2"] = a
            df["pred_log10[kcat]_final"] = p1
            df["pred_log10[Km]_final"] = km_val
            df["pred_log10[kcat/Km]_final"] = a

        else:
            sbt1 = np.stack(df["sbt_feat1"].values)
            sbt2 = np.stack(df["sbt_feat2"].values)
            sbt1 = th.tensor(sbt1, dtype=th.float32).to(DEVICE)
            sbt2 = th.tensor(sbt2, dtype=th.float32).to(DEVICE)

            p1_all, p2_all, km1_all, km2_all, a1_all, a2_all = [], [], [], [], [], []
            for kc, km, act in zip(kcat_list, km_list, act_list):
                pred1 = inference(kc, km, act, ezy_feat, sbt1, DEVICE)
                pred2 = inference(kc, km, act, ezy_feat, sbt2, DEVICE)
                p1_all.append(pred1[:, [0]])
                p2_all.append(pred2[:, [0]])
                km1_all.append(pred1[:, [1]])
                km2_all.append(pred2[:, [1]])
                a1_all.append(pred1[:, [2]])
                a2_all.append(pred2[:, [2]])

            p1 = np.mean(np.concatenate(p1_all, axis=1), axis=1)
            p2 = np.mean(np.concatenate(p2_all, axis=1), axis=1)
            km1 = np.mean(np.concatenate(km1_all, axis=1), axis=1)
            km2 = np.mean(np.concatenate(km2_all, axis=1), axis=1)
            a1 = np.mean(np.concatenate(a1_all, axis=1), axis=1)
            a2 = np.mean(np.concatenate(a2_all, axis=1), axis=1)

            df["pred_log10[kcat]_sub1"] = p1
            df["pred_log10[kcat]_sub2"] = p2
            df["pred_log10[Km]_sub1"] = km1
            df["pred_log10[Km]_sub2"] = km2
            df["pred_log10[kcat/Km]_sub1"] = a1
            df["pred_log10[kcat/Km]_sub2"] = a2
            df["pred_log10[kcat]_final"] = (p1 + p2) / 2
            df["pred_log10[Km]_final"] = (km1 + km2) / 2
            df["pred_log10[kcat/Km]_final"] = (a1 + a2) / 2

        df.to_pickle(out_path)
        print(f"✅ Done: {fname}")

    print("\n🎉 All finished!")