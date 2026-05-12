import pandas as pd
import re

# ===================== 读取数据 =====================
file_path = r"D:\fwq\catapro\datasets\kcat-data_0.4simi-10fold.csv"
df = pd.read_csv(file_path)

core_cols = ["EC", "EnzymeType", "UniProtID", "Substrate", "Smiles", "kcat(s^-1)", "fold"]
df = df[core_cols].dropna(subset=["UniProtID", "Smiles", "EnzymeType"]).copy()
df["enz_sub_combo"] = df["UniProtID"] + " + " + df["Substrate"]

# ===================== 突变判断规则 =====================
single_mut_pattern = re.compile(r'^([A-Z])(\d+)([A-Z])$')


def extract_mut_site(enz_type):
    match = single_mut_pattern.match(str(enz_type))
    return match.group(2) if match else None


# ===================== 按每个反应统计 =====================
result = []

for combo_name, group in df.groupby("enz_sub_combo"):
    types = group["EnzymeType"].tolist()
    wild = any("wild" in str(t).lower() for t in types)
    mutants = [t for t in types if not "wild" in str(t).lower()]
    mutant_set = set(mutants)

    single_muts = [m for m in mutant_set if single_mut_pattern.match(str(m))]
    multi_muts = [m for m in mutant_set if re.search(r'[A-Z]\d+[A-Z][\+/_\-][A-Z]\d+[A-Z]', str(m))]

    total_mut = len(mutant_set)
    total_single = len(single_muts)
    total_multi = len(multi_muts)

    # 同一位点不同突变
    mut_sites = [extract_mut_site(m) for m in single_muts]
    mut_sites = [s for s in mut_sites if s]
    has_same_site_mut = len(mut_sites) != len(set(mut_sites)) and len(mut_sites) >= 2

    # 分类
    if wild and total_single > 0 and total_multi > 0:
        cat = "野生型+单突变+多位点突变"
    elif wild and total_single > 0:
        cat = "野生型+单突变"
    elif wild and total_multi > 0:
        cat = "野生型+多位点突变"
    elif total_single > 0 and total_multi > 0:
        cat = "无野生型+单突变+多位点突变"
    elif wild:
        cat = "仅野生型"
    else:
        cat = "仅突变体（无野生型）"

    result.append({
        "分类": cat,
        "酶+底物": combo_name,
        "总突变数": total_mut,
        "单突变数": total_single,
        "多位点突变数": total_multi,
        "含同一位点不同突变": "是" if has_same_site_mut else "否"
    })

stat = pd.DataFrame(result)

# ===================== 🔥 剔除 仅野生型 / 总突变=0 =====================
stat = stat[stat["总突变数"] > 2].copy()

total = len(stat)

# ===================== 输出：按分类逐个列出每一个反应 =====================
print("=" * 120)
print("✅ 总反应数量（仅含突变）：", total)
print("=" * 120)

# 按分类输出，每个分类下列出所有反应
for category in stat["分类"].unique():
    sub_df = stat[stat["分类"] == category]

    print(f"\n🔹 【分类：{category}】  总数：{len(sub_df)} 个")
    print("-" * 120)
    print(f"{'UniProtID+底物':<50s} {'总突变':<6s} {'单突变':<6s} {'多位点':<6s} {'同一位点不同突变':<10s}")
    print("-" * 120)

    for _, row in sub_df.iterrows():
        print(
            f"{row['酶+底物']:<50s} {row['总突变数']:<6d} {row['单突变数']:<6d} {row['多位点突变数']:<6d} {row['含同一位点不同突变']:<10s}")