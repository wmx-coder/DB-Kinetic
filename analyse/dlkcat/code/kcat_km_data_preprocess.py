# 脚本顶部：正确导入（删除自导入，显式导入json）
import pandas as pd
import numpy as np
import json
import os
from rdkit import Chem  # 可选：用于验证SMILES合法性

# 1. 定义数据集路径
csv_path = "/mnt/usb1/wmx/catapro/datasets/kcat-over-Km-data_0.4simi-10fold.csv"

# 2. 读取CSV文件
df = pd.read_csv(csv_path)

# 3. 记录原始样本数
original_sample_count = len(df)
print("原始数据集形状：", df.shape)
print("原始样本总数：", original_sample_count)
print("原始fold列分布：")
print(df["fold"].value_counts().sort_index())

# ---------------------- 提前排查含"."的SMILES（可选） ----------------------
df["Smiles"] = df["Smiles"].astype(str)
# 提取含"."的SMILES样本并导出
smiles_with_dot = df[df["Smiles"].str.contains(".", na=False, regex=False)].copy()
smiles_dot_export_path = "smiles_with_dot.csv"
smiles_with_dot[["Smiles", "Sequence", "kcat(s^-1)", "Km(M)", "fold"]].to_csv(smiles_dot_export_path, index=False)
print(f"\n【SMILES排查】含'.'的SMILES样本数：{len(smiles_with_dot)}")
print(f"【SMILES排查】已导出至：{smiles_dot_export_path}")
print(f"【SMILES排查】前10个含'.'的SMILES：")
print(smiles_with_dot["Smiles"].head(10).tolist())

# 验证SMILES合法性（可选）
def is_valid_single_molecule_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and "." not in smiles
    except:
        return False

smiles_with_dot["is_valid_single_mol"] = smiles_with_dot["Smiles"].apply(is_valid_single_molecule_smiles)
valid_count = smiles_with_dot["is_valid_single_mol"].sum()
print(f"【SMILES排查】含'.'的SMILES中有效单分子数量：{valid_count}")
if valid_count > 0:
    smiles_with_dot[smiles_with_dot["is_valid_single_mol"]].to_csv("valid_smiles_with_dot.csv", index=False)
    print(f"【SMILES排查】有效单分子已导出至：valid_smiles_with_dot.csv")

# ---------------------- 分步统计剔除样本数 ----------------------
# 步骤1：剔除kcat≤0、Km无效（≤0/空/NaN）的样本，统计该步骤剔除数量
kcat_km_before = len(df)
# 筛选条件：kcat>0 且 Km>0 且 Km非空非NaN
valid_kcat_km = (df["kcat(s^-1)"] > 0) & \
                (df["Km(M)"].notna()) & \
                (df["Km(M)"] > 0)
df = df[valid_kcat_km].reset_index(drop=True)
kcat_km_after = len(df)
kcat_km_invalid_count = kcat_km_before - kcat_km_after
print(f"\n【步骤1】剔除kcat≤0 或 Km无效（≤0/空/NaN）的样本数：{kcat_km_invalid_count}")
print(f"步骤1后样本数：{kcat_km_after}")

# 新增：计算kcat/km列（核心改动，规避除零错误，已提前筛选有效Km）
df["kcat_km(s^-1·M^-1)"] = df["kcat(s^-1)"] / df["Km(M)"]
print(f"\n【核心计算】已完成kcat/km计算，列名：kcat_km(s^-1·M^-1)")
print(f"kcat/km前5个值：{df['kcat_km(s^-1·M^-1)'].head().tolist()}")

# 步骤2：剔除SMILES含"."的样本（修复筛选逻辑）
smiles_before = len(df)
df["Smiles"] = df["Smiles"].astype(str)
df = df[~df["Smiles"].str.contains(".", na=False, regex=False)].reset_index(drop=True)
smiles_after = len(df)
smiles_invalid_count = smiles_before - smiles_after
print(f"\n【步骤2】剔除SMILES含'.'的样本数：{smiles_invalid_count}")
print(f"步骤2后样本数：{smiles_after}")

# 步骤3：剔除蛋白质序列无效的样本，统计该步骤剔除数量
seq_before = len(df)
df["Sequence"] = df["Sequence"].astype(str)
df = df[df["Sequence"].notna() & (df["Sequence"] != "") & (df["Sequence"] != "nan")].reset_index(drop=True)
seq_after = len(df)
seq_invalid_count = seq_before - seq_after
print(f"\n【步骤3】剔除蛋白质序列无效的样本数：{seq_invalid_count}")
print(f"步骤3后样本数：{seq_after}")

# 步骤4：fold列转为int型并筛选0-9，统计该步骤剔除数量
fold_before = len(df)
df["fold"] = df["fold"].astype(int)
df = df[df["fold"].between(0, 9)].reset_index(drop=True)
fold_after = len(df)
fold_invalid_count = fold_before - fold_after
print(f"\n【步骤4】剔除fold不在0-9的样本数：{fold_invalid_count}")
print(f"步骤4后样本数：{fold_after}")

# ---------------------- 汇总剔除信息 ----------------------
total_invalid_count = original_sample_count - len(df)
print(f"\n==================== 清洗汇总 ====================")
print(f"原始样本总数：{original_sample_count}")
print(f"清洗后样本总数：{len(df)}")
print(f"总剔除样本数：{total_invalid_count}")
print(f"其中：")
print(f"  - kcat≤0 或 Km无效（≤0/空/NaN）的样本：{kcat_km_invalid_count} 个")
print(f"  - SMILES含'.'的样本：{smiles_invalid_count} 个")
print(f"  - 蛋白质序列无效的样本：{seq_invalid_count} 个")
print(f"  - fold异常的样本：{fold_invalid_count} 个")
print(f"===================================================")

# 5. 打印清洗后fold分布
print(f"\n清洗后fold列分布：")
print(df["fold"].value_counts().sort_index())

# ---------------------- 后续逻辑（修复JSON保存，适配kcat/km） ----------------------
# 1. 重命名kcat/km列为DLKcat标准名称"Value"（核心改动：替换原kcat(s^-1)）
df.rename(
    columns={"kcat_km(s^-1·M^-1)": "Value"},
    inplace=True
)

# 2. 仅保留核心字段（Smiles、Sequence、Value、fold），减少冗余（不保留额外列）
core_df = df[["Smiles", "Sequence", "Value", "fold"]].copy()

# 3. 最终确认核心数据
print("\n核心字段整理完成，列名：", core_df.columns.tolist())
print("核心数据前5行：")
print(core_df.head())
print(f"核心数据样本数：{len(core_df)}")

# 1. 转换为DLKcat要求的JSON格式（仅保留Smiles、Sequence、Value）
kcat_json = core_df[["Smiles", "Sequence", "Value"]].to_dict("records")

# 2. 保存JSON文件（先创建目录，再保存）
json_save_path = "../../Data/database/Kcat_km_combination_0918.json"
json_dir = os.path.dirname(json_save_path)
os.makedirs(json_dir, exist_ok=True)
with open(json_save_path, "w") as f:
    json.dump(kcat_json, f, indent=2)

print(f"\nJSON文件已保存至：{json_save_path}")
print(f"JSON文件包含样本数：{len(kcat_json)}")
print(f"JSON数据前1条示例：{kcat_json[0]}")

# 提取fold列，确保顺序与JSON文件（core_df）完全一致
folds = core_df["fold"].values  # 一维int数组，shape=[N,]

# 保存folds.npy（先创建目录，再保存）
folds_save_path = "../../Data/kcat_km/input/folds.npy"
folds_dir = os.path.dirname(folds_save_path)
os.makedirs(folds_dir, exist_ok=True)
np.save(folds_save_path, folds)

print(f"\nfolds.npy已保存至：{folds_save_path}")
print(f"folds.npy形状：{folds.shape}")
print(f"folds.npy取值范围：{folds.min()} - {folds.max()}")
print(f"folds.npy各折样本数：")
for fold in range(10):
    print(f"折{fold}：{sum(folds == fold)}个样本")