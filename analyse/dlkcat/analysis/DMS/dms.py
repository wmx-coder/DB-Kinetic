# import pandas as pd
# import os
# import numpy as np
#
# # ===================== 路径配置 =====================
# RAW_CSV_DIR = "/mnt/usb1/wmx/catapro/analyse/dms/"
# PRED_TSV_DIR = "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/"
# OUTPUT_DIR = "/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/"
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # 原始CSV → 预测TSV的前缀 正确对应
# DATASETS = {
#     "EcTL_with_sub.csv": "EcTL",
#     "HIS3.csv": "HIS3",
#     "HIS7.csv": "HIS7",
#     "si-dbs_ub.csv": "si-dbs_ub",
#     "Ssdata_ub.csv": "Ssdata_ub",
#     "Ttdata_with_sub.csv": "Ttdata"
# }
#
# # ===================== 按序列精确匹配 =====================
# for csv_file, pred_prefix in DATASETS.items():
#     print(f"\n===== 处理：{csv_file} =====")
#
#     # 1. 读取原始 CSV
#     raw_path = os.path.join(RAW_CSV_DIR, csv_file)
#     df_raw = pd.read_csv(raw_path)
#
#     # 自动识别序列列
#     seq_col = "Sequence" if "Sequence" in df_raw.columns else "sequence"
#
#     # 2. 读取底物1 预测（正确文件名）
#     tsv1 = os.path.join(PRED_TSV_DIR, f"{pred_prefix}_sbt1_pred.tsv")
#     df1 = pd.read_csv(tsv1, sep="\t")
#     df1 = df1[["Sequence", "Pred_log_avg", "Pred_value_avg"]].rename(columns={
#         "Pred_log_avg": "pred_log_sub1",
#         "Pred_value_avg": "pred_val_sub1"
#     })
#
#     # 3. 读取底物2 预测（正确文件名）
#     tsv2 = os.path.join(PRED_TSV_DIR, f"{pred_prefix}_sbt2_pred.tsv")
#     df2 = pd.read_csv(tsv2, sep="\t")
#     df2 = df2[["Sequence", "Pred_log_avg", "Pred_value_avg"]].rename(columns={
#         "Pred_log_avg": "pred_log_sub2",
#         "Pred_value_avg": "pred_val_sub2"
#     })
#
#     # ----------------- ✅ 按序列匹配 -----------------
#     df_merge = df_raw.merge(df1, left_on=seq_col, right_on="Sequence", how="left")
#     df_merge = df_merge.merge(df2, left_on=seq_col, right_on="Sequence", how="left")
#
#     # 去重
#     df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
#
#     # 4. 计算 final
#     df_merge["pred_log_final"] = (df_merge["pred_log_sub1"] + df_merge["pred_log_sub2"]) / 2
#     df_merge["pred_val_final"] = (df_merge["pred_val_sub1"] + df_merge["pred_val_sub2"]) / 2
#
#     # 5. 保存
#     out_name = csv_file.replace(".csv", "_dlkcat.csv")
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     df_merge.to_csv(out_path, index=False)
#     print(f"✅ 合并完成：{out_path}")
#
# print("\n🎉🎉🎉 全部合并完成！")

import pandas as pd
import os
import numpy as np

# ===================== 路径配置 =====================
RAW_CSV_DIR    = "/mnt/usb1/wmx/catapro/analyse/dms/"
PRED_TSV_DIR   = "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/"
OUTPUT_DIR     = "/mnt/usb1/wmx/catapro/analyse/dms/dlkcat_merge_result/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 只处理这 3 个单底物
DATASETS = {
    "HIS3.csv": "HIS3",
    "HIS7.csv": "HIS7",
    "si-dbs_ub.csv": "si-dbs_ub"
}

# ===================== 单底物合并 =====================
for csv_file, pred_prefix in DATASETS.items():
    print(f"\n===== 处理单底物：{csv_file} =====")

    # 1. 原始数据
    raw_path = os.path.join(RAW_CSV_DIR, csv_file)
    df_raw = pd.read_csv(raw_path)
    seq_col = "Sequence" if "Sequence" in df_raw.columns else "sequence"

    # 2. 读取唯一的预测文件
    tsv_path = os.path.join(PRED_TSV_DIR, f"{pred_prefix}_pred.tsv")
    df_pred = pd.read_csv(tsv_path, sep="\t")

    # 严格使用你说的列名：Pred_log_avg, Pred_value_avg
    df_pred = df_pred.rename(columns={
        "Pred_log_avg":    "pred_log_sub1",
        "Pred_value_avg":  "pred_val_sub1"
    })

    # 3. 按序列合并
    df_merge = df_raw.merge(
        df_pred[["Sequence", "pred_log_sub1", "pred_val_sub1"]],
        left_on=seq_col,
        right_on="Sequence",
        how="left"
    )

    # ===================== 单底物：三列完全一样 =====================
    df_merge["pred_log_sub2"]    = df_merge["pred_log_sub1"]
    df_merge["pred_val_sub2"]    = df_merge["pred_val_sub1"]
    df_merge["pred_log_final"]   = df_merge["pred_log_sub1"]
    df_merge["pred_val_final"]   = df_merge["pred_val_sub1"]

    # 去重
    df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]

    # 保存
    out_name = csv_file.replace(".csv", "_dlkcat.csv")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    df_merge.to_csv(out_path, index=False)

    print(f"✅ 生成列：pred_log_sub1, pred_log_sub2, pred_log_final")
    print(f"✅ 合并完成：{out_path}")

print("\n🎉🎉🎉 3 个单底物全部合并完成！")