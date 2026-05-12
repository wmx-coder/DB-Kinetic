import pandas as pd
import re

# ===================== 读取原始数据（所有列） =====================
df = pd.read_pickle("/mnt/usb3/wmx/kcat_km/kcat_km_with_log_feats.pkl")
df = df.dropna(subset=["UniProtID", "Smiles", "EnzymeType"]).copy()

# ===================== 突变判断（你的格式） =====================
def is_mut(enz):
    s = str(enz).strip().lower()
    if "wild" in s:
        return False
    return re.match(r'^[A-Z]\d+[A-Z](\/[A-Z]\d+[A-Z])*$', str(enz).strip()) is not None

# ===================== 找出【有突变的反应】 =====================
df["group_key"] = df["UniProtID"].astype(str) + "_|_" + df["Smiles"].astype(str)
valid_groups = set()

for g, grp in df.groupby("group_key"):
    for et in grp["EnzymeType"]:
        if is_mut(et):
            valid_groups.add(g)
            break

# ===================== 筛选 + 删除临时列 =====================
df_save = df[df["group_key"].isin(valid_groups)].copy()
df_save = df_save.drop(columns=["group_key"])

# ===================== 保存（完整原始数据） =====================
df_save.to_pickle("/mnt/usb3/wmx/kcat_km/case/kcat_km_final_valid_samples.pkl")