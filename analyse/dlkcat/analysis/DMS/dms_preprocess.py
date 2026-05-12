import pandas as pd
import numpy as np
import json
import os

# ==============================================
# 【配置：6 个 DMS 数据集】
# ==============================================
DATASETS = {
    "EcTL":      "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub.csv",
    "HIS3":      "/mnt/usb1/wmx/catapro/analyse/dms/HIS3.csv",
    "HIS7":      "/mnt/usb1/wmx/catapro/analyse/dms/HIS7.csv",
    "si-dbs_ub": "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub.csv",
    "Ssdata_ub": "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub.csv",
    "Ttdata":    "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub.csv"
}

# 输出根目录
OUTPUT_ROOT = "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ==============================================
# 【核心函数：自动拆分双底物 reactant_smiles → sbt1 + sbt2】
# ==============================================
def clean_split_bisubstrate(csv_path, name):
    print(f"\n{'='*60}")
    print(f" 处理：{name} | 自动拆分双底物")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"原始样本数: {len(df)}")

    # ----------------------
    # 自动识别列
    # ----------------------
    smi_col = "reactant_smiles"
    seq_col = "Sequence" if "Sequence" in df.columns else "sequence"

    # ----------------------
    # 按 . 拆分双底物
    # ----------------------
    df = df[df[smi_col].notna()].copy()
    df["split"] = df[smi_col].str.split("\.")
    df = df[df["split"].str.len() >= 2].copy()  # 必须能拆成 2 个
    df["sbt1"] = df["split"].str[0].str.strip()
    df["sbt2"] = df["split"].str[1].str.strip()

    print(f"拆分后有效样本数: {len(df)}")

    # ----------------------
    # 清洗（和你原来逻辑一致）
    # ----------------------
    def clean_pair(df, sbt_name):
        d = df[[sbt_name, seq_col]].copy()
        d.columns = ["Smiles", "Sequence"]
        d = d[d["Smiles"].notna()]
        d = d[d["Sequence"].notna()]
        d = d[~d["Smiles"].str.contains(r"^\s*$")]
        d = d[~d["Sequence"].str.contains(r"^\s*$")]
        return d.to_dict("records")

    # 生成两个底物 JSON
    json1 = clean_pair(df, "sbt1")
    json2 = clean_pair(df, "sbt2")

    return json1, json2, len(json1)

# ==============================================
# 批量处理 6 个文件
# ==============================================
if __name__ == "__main__":
    for name, path in DATASETS.items():
        j1, j2, cnt = clean_split_bisubstrate(path, name)

        # 保存 sbt1
        p1 = os.path.join(OUTPUT_ROOT, f"{name}_sbt1.json")
        with open(p1, "w") as f:
            json.dump(j1, f, indent=2)

        # 保存 sbt2
        p2 = os.path.join(OUTPUT_ROOT, f"{name}_sbt2.json")
        with open(p2, "w") as f:
            json.dump(j2, f, indent=2)

        print(f"✅ {name} 完成 | 样本数: {cnt}")
        print(f"   底物1 → {p1}")
        print(f"   底物2 → {p2}")

    print("\n🎉🎉🎉 6 个数据集 双底物拆分 + JSON 全部完成！")