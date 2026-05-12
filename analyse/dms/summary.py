# # import pandas as pd
# # from scipy.stats import pearsonr
# # import numpy as np
# #
# # def safe_scc(a, b):
# #     # 自动去掉 NaN 和 inf，保证计算安全
# #     df = pd.DataFrame({'x': a, 'y': b}).dropna()
# #     df = df[np.isfinite(df).all(1)]
# #     return pearsonr(df.x, df.y)[0]
# #
# # # ===================== EcTL =====================
# # print("===== EcTL =====")
# # df = pd.read_csv("/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/EcTL_with_sub_dlkcat.csv")
# # print(f"DLKCat | sub1: {safe_scc(df.log10kcat_max, df.pred_log_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/EcTL_eitlem_kcat.pkl")
# # print(f"Eitlem | sub1: {safe_scc(df.log10kcat_max, df.pred_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_kcat_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats_pred.pkl")
# # print(f"CataPro| sub1: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub1']):.4f} | sub2: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub2']):.4f} | final: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_final']):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats_pred_my.pkl")
# # print(f"MyModel| sub1: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_final):.4f}")
# #
# # # ===================== HIS3 =====================
# # print("\n===== HIS3 =====")
# # df = pd.read_csv("/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/HIS3_dlkcat.csv")
# # print(f"DLKCat | sub1: {safe_scc(df.log10kcat_max, df.pred_log_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/HIS3_eitlem_kcat.pkl")
# # print(f"Eitlem | sub1: {safe_scc(df.log10kcat_max, df.pred_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_kcat_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats_pred.pkl")
# # print(f"CataPro| sub1: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub1']):.4f} | sub2: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub2']):.4f} | final: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_final']):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats_pred_my.pkl")
# # print(f"MyModel| sub1: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_final):.4f}")
# #
# # # ===================== HIS7 =====================
# # print("\n===== HIS7 =====")
# # df = pd.read_csv("/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/HIS7_dlkcat.csv")
# # print(f"DLKCat | sub1: {safe_scc(df.log10kcat_max, df.pred_log_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/HIS7_eitlem_kcat.pkl")
# # print(f"Eitlem | sub1: {safe_scc(df.log10kcat_max, df.pred_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_kcat_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats_pred.pkl")
# # print(f"CataPro| sub1: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub1']):.4f} | sub2: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub2']):.4f} | final: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_final']):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats_pred_my.pkl")
# # print(f"MyModel| sub1: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_final):.4f}")
# #
# # # ===================== si-dbs_ub =====================
# # print("\n===== si-dbs_ub =====")
# # df = pd.read_csv("/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/si-dbs_ub_dlkcat.csv")
# # print(f"DLKCat | sub1: {safe_scc(df.log10kcat_max, df.pred_log_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_eitlem_kcat.pkl")
# # print(f"Eitlem | sub1: {safe_scc(df.log10kcat_max, df.pred_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_kcat_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats_pred.pkl")
# # print(f"CataPro| sub1: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub1']):.4f} | sub2: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub2']):.4f} | final: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_final']):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats_pred_my.pkl")
# # print(f"MyModel| sub1: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_final):.4f}")
# #
# # # ===================== Ssdata_ub =====================
# # print("\n===== Ssdata_ub =====")
# # df = pd.read_csv("/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/Ssdata_ub_dlkcat.csv")
# # print(f"DLKCat | sub1: {safe_scc(df.log10kcat_max, df.pred_log_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_eitlem_kcat.pkl")
# # print(f"Eitlem | sub1: {safe_scc(df.log10kcat_max, df.pred_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_kcat_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats_pred.pkl")
# # print(f"CataPro| sub1: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub1']):.4f} | sub2: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub2']):.4f} | final: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_final']):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats_pred_my.pkl")
# # print(f"MyModel| sub1: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_final):.4f}")
# #
# # # ===================== Ttdata =====================
# # print("\n===== Ttdata =====")
# # df = pd.read_csv("/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/Ttdata_with_sub_dlkcat.csv")
# # print(f"DLKCat | sub1: {safe_scc(df.log10kcat_max, df.pred_log_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_eitlem_kcat.pkl")
# # print(f"Eitlem | sub1: {safe_scc(df.log10kcat_max, df.pred_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_kcat_final):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats_pred.pkl")
# # print(f"CataPro| sub1: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub1']):.4f} | sub2: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_sub2']):.4f} | final: {safe_scc(df.log10kcat_max, df['pred_log10[kcat]_final']):.4f}")
# #
# # df = pd.read_pickle("/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats_pred_my.pkl")
# # print(f"MyModel| sub1: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub1):.4f} | sub2: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_sub2):.4f} | final: {safe_scc(df.log10kcat_max, df.pred_log10_kcat_final):.4f}")
# #
#
# import pandas as pd
# import numpy as np
# from scipy.stats import spearmanr
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # ===================== 路径 =====================
# base = "/mnt/usb1/wmx/catapro/analyse/dms"
# files = {
#     "CataPro": f"{base}/EcTL_with_sub_feats_pred.pkl",
#     "DB-Kinetic_kcat": f"{base}/EcTL_with_sub_feats_pred_my.pkl",
#     "DB-Kinetic_kcatkm": f"{base}/kcat_km/EcTL_with_sub_feats_pred.pkl",
#     "DB-Kinetic_km": f"{base}/km/EcTL_with_sub_feats_catapro.pkl",
#     "Eitlem_Kcat_Km": f"{base}/EcTL_Kcat_Km.pkl",
#     "Eitlem_KKM": f"{base}/EcTL_KKM_PRED.pkl",
# }
#
# # 真实标签
# true_col = "log10kcat_max"
#
# # ===================== 读取所有数据 =====================
# dfs = {}
# for name, path in files.items():
#     dfs[name] = pd.read_pickle(path)
#     print(f"✅ 读取：{name}")
#
# # ===================== 定义要计算的 (模型, 任务, 列名) =====================
# comparisons = [
#     # CataPro
#     ("CataPro", "kcat", "pred_log10[kcat]_sub1"),
#     ("CataPro", "kcat", "pred_log10[kcat]_sub2"),
#     ("CataPro", "kcat", "pred_log10[kcat]_final"),
#
#     ("CataPro", "Km", "pred_log10[Km]_sub1"),
#     ("CataPro", "Km", "pred_log10[Km]_sub2"),
#     ("CataPro", "Km", "pred_log10[Km]_final"),
#
#     ("CataPro", "kcat/Km", "pred_log10[kcat/Km]_sub1"),
#     ("CataPro", "kcat/Km", "pred_log10[kcat/Km]_sub2"),
#     ("CataPro", "kcat/Km", "pred_log10[kcat/Km]_final"),
#
#     # 你的 DB-Kinetic
#     ("DB-Kinetic", "kcat", "pred_log10_kcat_sub1"),
#     ("DB-Kinetic", "kcat", "pred_log10_kcat_sub2"),
#     ("DB-Kinetic", "kcat", "pred_log10_kcat_final"),
#
#     ("DB-Kinetic", "Km", "pred_log10[Km(mM)]_sub1"),
#     ("DB-Kinetic", "Km", "pred_log10[Km(mM)]_sub2"),
#     ("DB-Kinetic", "Km", "pred_log10[Km(mM)]_final"),
#
#     ("DB-Kinetic", "kcat/Km", "pred_fusion_log10[kcat/Km]_sub1"),
#     ("DB-Kinetic", "kcat/Km", "pred_fusion_log10[kcat/Km]_sub2"),
#     ("DB-Kinetic", "kcat/Km", "pred_fusion_log10[kcat/Km]_final"),
#
#     # Eitlem
#     ("Eitlem", "kcat", "pred_kcat_sub1"),
#     ("Eitlem", "kcat", "pred_kcat_sub2"),
#     ("Eitlem", "kcat", "pred_kcat_final"),
#
#     ("Eitlem", "Km", "pred_km_sub1"),
#     ("Eitlem", "Km", "pred_km_sub2"),
#     ("Eitlem", "Km", "pred_km_final"),
#
#     ("Eitlem", "kcat/Km", "pred_kkm_sub1"),
#     ("Eitlem", "kcat/Km", "pred_kkm_sub2"),
#     ("Eitlem", "kcat/Km", "pred_kkm_final"),
# ]
#
# # ===================== 计算 SCC =====================
# results = []
# for model, task, pred_col in comparisons:
#     # 找到对应 df
#     if model == "CataPro":
#         df = dfs["CataPro"]
#     elif model == "DB-Kinetic":
#         if task == "kcat":
#             df = dfs["DB-Kinetic_kcat"]
#         elif task == "Km":
#             df = dfs["DB-Kinetic_km"]
#         else:
#             df = dfs["DB-Kinetic_kcatkm"]
#     elif model == "Eitlem":
#         if task in ["kcat", "Km"]:
#             df = dfs["Eitlem_Kcat_Km"]
#         else:
#             df = dfs["Eitlem_KKM"]
#
#     y_true = df[true_col].values
#     y_pred = df[pred_col].values
#
#     # 过滤 NaN
#     mask = ~(np.isnan(y_true) | np.isnan(y_pred))
#     scc = spearmanr(y_true[mask], y_pred[mask])[0]
#
#     suffix = "sub1" if "sub1" in pred_col else "sub2" if "sub2" in pred_col else "final"
#     results.append({
#         "Model": model,
#         "Task": task,
#         "Type": suffix,
#         "SCC": round(scc, 4)
#     })
#
# # ===================== 输出结果 =====================
# df_res = pd.DataFrame(results)
# print("\n" + "=" * 80)
# print("📊 EcTL 数据集 - 所有模型 SCC 指标（log10kcat_max 为标签）")
# print("=" * 80)
# print(df_res.to_string(index=False))
#
# # ===================== 保存为 Excel =====================
# out_path = f"{base}/EcTL_SCC_results.xlsx"
# df_res.to_excel(out_path, index=False)
# print(f"\n🎉 结果已保存：{out_path}")

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

true_col = "log10kcat_max"

# ========== 所有文件 全路径写死（100% 按你给的路径） ==========
all_files = {
    # ==================== EcTL ====================
    "EcTL": {
        "CataPro":              "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats_pred.pkl",
        "DB-Kinetic_kcat":      "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats_pred_my.pkl",
        "DB-Kinetic_kcatkm":    "/mnt/usb1/wmx/catapro/analyse/dms/kcat_km/EcTL_with_sub_feats_pred.pkl",
        "DB-Kinetic_km":        "/mnt/usb1/wmx/catapro/analyse/dms/km/EcTL_with_sub_feats_catapro.pkl",
        "Eitlem_Kcat_Km":       "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_Kcat_Km.pkl",
        "Eitlem_KKM":           "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_KKM_PRED.pkl",
    },
    # ==================== HIS3 ====================
    "HIS3": {
        "CataPro":              "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats_pred.pkl",
        "DB-Kinetic_kcat":      "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats_pred_my.pkl",
        "DB-Kinetic_kcatkm":    "/mnt/usb1/wmx/catapro/analyse/dms/kcat_km/HIS3_feats_pred.pkl",
        "DB-Kinetic_km":        "/mnt/usb1/wmx/catapro/analyse/dms/km/HIS3_feats_catapro.pkl",
        "Eitlem_Kcat_Km":       "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_Kcat_Km.pkl",
        "Eitlem_KKM":           "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_KKM_PRED.pkl",
    },
    # ==================== HIS7 ====================
    "HIS7": {
        "CataPro":              "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats_pred.pkl",
        "DB-Kinetic_kcat":      "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats_pred_my.pkl",
        "DB-Kinetic_kcatkm":    "/mnt/usb1/wmx/catapro/analyse/dms/kcat_km/HIS7_feats_pred.pkl",
        "DB-Kinetic_km":        "/mnt/usb1/wmx/catapro/analyse/dms/km/HIS7_feats_catapro.pkl",
        "Eitlem_Kcat_Km":       "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_Kcat_Km.pkl",
        "Eitlem_KKM":           "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_KKM_PRED.pkl",
    },
    # ==================== si-dbs_ub ====================
    "si-dbs_ub": {
        "CataPro":              "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats_pred.pkl",
        "DB-Kinetic_kcat":      "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats_pred_my.pkl",
        "DB-Kinetic_kcatkm":    "/mnt/usb1/wmx/catapro/analyse/dms/kcat_km/si-dbs_ub_feats_pred.pkl",
        "DB-Kinetic_km":        "/mnt/usb1/wmx/catapro/analyse/dms/km/si-dbs_ub_feats_catapro.pkl",
        "Eitlem_Kcat_Km":       "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_Kcat_Km.pkl",
        "Eitlem_KKM":           "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_KKM_PRED.pkl",
    },
    # ==================== Ssdata_ub ====================
    "Ssdata_ub": {
        "CataPro":              "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats_pred.pkl",
        "DB-Kinetic_kcat":      "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats_pred_my.pkl",
        "DB-Kinetic_kcatkm":    "/mnt/usb1/wmx/catapro/analyse/dms/kcat_km/Ssdata_ub_feats_pred.pkl",
        "DB-Kinetic_km":        "/mnt/usb1/wmx/catapro/analyse/dms/km/Ssdata_ub_feats_catapro.pkl",
        "Eitlem_Kcat_Km":       "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_Kcat_Km.pkl",
        "Eitlem_KKM":           "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_KKM_PRED.pkl",
    },
    # ==================== Ttdata ====================
    "Ttdata": {
        "CataPro":              "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats_pred.pkl",
        "DB-Kinetic_kcat":      "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats_pred_my.pkl",
        "DB-Kinetic_kcatkm":    "/mnt/usb1/wmx/catapro/analyse/dms/kcat_km/Ttdata_with_sub_feats_pred.pkl",
        "DB-Kinetic_km":        "/mnt/usb1/wmx/catapro/analyse/dms/km/Ttdata_with_sub_feats_catapro.pkl",
        "Eitlem_Kcat_Km":       "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_Kcat_Km.pkl",
        "Eitlem_KKM":           "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_KKM_PRED.pkl",
    },
}

# ========== 列名映射（完全按你给的） ==========
col_map = {
    "CataPro": {
        "kcat":     ["pred_log10[kcat]_sub1",    "pred_log10[kcat]_sub2",    "pred_log10[kcat]_final"],
        "Km":       ["pred_log10[Km]_sub1",      "pred_log10[Km]_sub2",      "pred_log10[Km]_final"],
        "kcat/Km":  ["pred_log10[kcat/Km]_sub1", "pred_log10[kcat/Km]_sub2", "pred_log10[kcat/Km]_final"]
    },
    "DB-Kinetic": {
        "kcat":     ["pred_log10_kcat_sub1",       "pred_log10_kcat_sub2",       "pred_log10_kcat_final"],
        "Km":       ["pred_log10[Km(mM)]_sub1",    "pred_log10[Km(mM)]_sub2",    "pred_log10[Km(mM)]_final"],
        "kcat/Km":  ["pred_fusion_log10[kcat/Km]_sub1", "pred_fusion_log10[kcat/Km]_sub2", "pred_fusion_log10[kcat/Km]_final"]
    },
    "Eitlem": {
        "kcat":     ["pred_kcat_sub1",         "pred_kcat_sub2",         "pred_kcat_final"],
        "Km":       ["pred_km_sub1",           "pred_km_sub2",           "pred_km_final"],
        "kcat/Km":  ["pred_kkm_sub1",          "pred_kkm_sub2",          "pred_kkm_final"]
    }
}

type_list = ["sub1", "sub2", "final"]
total_res = []

# ========== 开始计算 ==========
for ds_name, paths in all_files.items():
    print(f"\n========== 计算 {ds_name} ==========")
    dfs = {key: pd.read_pickle(path) for key, path in paths.items()}

    for model in ["CataPro", "DB-Kinetic", "Eitlem"]:
        for task in ["kcat", "Km", "kcat/Km"]:
            pred_cols = col_map[model][task]

            for idx, pred_col in enumerate(pred_cols):
                subtype = type_list[idx]

                # 选择对应数据框
                if model == "CataPro":
                    df = dfs["CataPro"]
                elif model == "DB-Kinetic":
                    if task == "kcat":
                        df = dfs["DB-Kinetic_kcat"]
                    elif task == "Km":
                        df = dfs["DB-Kinetic_km"]
                    else:
                        df = dfs["DB-Kinetic_kcatkm"]
                else:
                    if task in ["kcat", "Km"]:
                        df = dfs["Eitlem_Kcat_Km"]
                    else:
                        df = dfs["Eitlem_KKM"]

                # 计算 SCC
                y_true = df[true_col].values
                y_pred = df[pred_col].values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                scc = spearmanr(y_true[mask], y_pred[mask])[0]

                total_res.append({
                    "Dataset": ds_name,
                    "Model": model,
                    "Task": task,
                    "Type": subtype,
                    "SCC": round(scc, 4)
                })

# ========== 保存结果 ==========
res_df = pd.DataFrame(total_res)
out_path = "/mnt/usb1/wmx/catapro/analyse/dms/ALL_DATASET_SCC_FINAL.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    res_df.to_excel(writer, sheet_name="ALL_RESULTS", index=False)
    for ds in all_files.keys():
        res_df[res_df["Dataset"] == ds].to_excel(writer, sheet_name=ds, index=False)

# ========== 输出展示 ==========
print("\n" * 2)
print("=" * 80)
print("          所有数据集 SCC 计算完成 ✅")
print("=" * 80)
print(res_df.to_string(index=False))
print(f"\n文件已保存到：\n{out_path}")