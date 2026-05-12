import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ===================== 仅读取你的预测文件 =====================
df = pd.read_pickle("/mnt/usb3/wmx/kcat/case/predict.pkl")

# 自动生成 酶+底物 组合键
df["enz_sub_combo"] = df["UniProtID"].astype(str) + " | " + df["Substrate"].astype(str)

# ===================== 突变判断规则 =====================
single_mut_pattern = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def extract_mut_site(enz_type):
    match = single_mut_pattern.match(str(enz_type))
    return match.group(2) if match else None

# ===================== 按酶+底物分组统计 =====================
result = []
for combo_name, group in df.groupby("enz_sub_combo"):
    types = group["EnzymeType"].tolist()
    ec = group["EC"].iloc[0]

    # 区分野生型 / 突变体
    has_wild = any("wild" in str(t).lower() for t in types)
    mutants = [t for t in types if not "wild" in str(t).lower()]
    mutant_set = list(set(mutants))
    total_mut = len(mutant_set)

    # 单突变 / 多位点突变
    single_muts = [m for m in mutant_set if single_mut_pattern.match(str(m))]
    multi_muts = [m for m in mutant_set if re.search(r'[A-Z]\d+[A-Z][\+/_\-][A-Z]\d+[A-Z]', str(m))]

    # 同一位点不同突变
    mut_sites = [extract_mut_site(m) for m in single_muts]
    mut_sites = [s for s in mut_sites if s]
    has_same_site_mut = len(mut_sites) != len(set(mut_sites)) and len(mut_sites) >= 2
    has_multi_mut = len(multi_muts) > 0

    # 分类
    if has_wild and len(single_muts) > 0 and len(multi_muts) > 0:
        cat = "野生型+单突变+多位点突变"
    elif has_wild and len(single_muts) > 0:
        cat = "野生型+单突变"
    elif has_wild and len(multi_muts) > 0:
        cat = "野生型+多位点突变"
    elif len(single_muts) > 0 and len(multi_muts) > 0:
        cat = "无野生型+单突变+多位点突变"
    else:
        cat = "仅突变体（无野生型）"

    # 计算 SCC PCC
    y_true = group["log10_kcat"].values
    y_pred = group["pred_log10_kcat"].values
    if len(y_true) >= 2:
        scc, _ = spearmanr(y_true, y_pred)
        pcc, _ = pearsonr(y_true, y_pred)
    else:
        scc, pcc = np.nan, np.nan

    result.append({
        "分类": cat,
        "EC": ec,
        "酶+底物": combo_name,
        "总突变数": total_mut,
        "含多位点突变": "是" if has_multi_mut else "否",
        "含同一位点突变": "是" if has_same_site_mut else "否",
        "SCC": round(scc, 4) if not np.isnan(scc) else None,
        "PCC": round(pcc, 4) if not np.isnan(pcc) else None,
    })

# ===================== 基础过滤 =====================
stat = pd.DataFrame(result)
stat = stat[(stat["总突变数"] >= 3) & (stat["总突变数"] <= 6)]
stat = stat[stat["SCC"] == 1.0]
stat = stat.dropna(subset=["PCC"])

# ===================== 按你的规则选16组 =====================
# 1. 先取突变数 4、5 且 PCC≥0.9 的全部
group45 = stat[(stat["总突变数"].isin([4,5])) & (stat["PCC"] >= 0.9)].copy()

# 2. 再取突变数=3，按PCC从高到低
group3 = stat[stat["总突变数"] == 3].copy()
group3 = group3.sort_values("PCC", ascending=False)

# 3. 合并取16组
top16_groups = pd.concat([group45, group3]).head(16)
selected_combos = top16_groups["酶+底物"].unique()

# ===================== 🔥 提取原始样本（真正的每条数据） =====================
df_selected = df[df["enz_sub_combo"].isin(selected_combos)].copy()

# 只保留你需要的列
keep_cols = [
    "UniProtID", "EC", "Smiles",
    "log10_kcat", "pred_log10_kcat",
    "EnzymeType", "Substrate"
]
df_out = df_selected[keep_cols].copy()

# ===================== 保存 =====================
save_path = "/mnt/usb3/wmx/kcat/case/top16_selected_samples.csv"
df_out.to_csv(save_path, index=False, encoding="utf-8-sig")

# ===================== 展示 =====================
print("\n" + "="*120)
print("✅ 已选出 16 组对应的原始样本，并保存为 CSV")
print(f"📁 保存路径：{save_path}")
print(f"📊 共 {len(df_out)} 条样本")
print("="*120)
print(df_out.head())