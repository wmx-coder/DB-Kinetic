import pandas as pd
from Bio import SeqIO
import os
from collections import defaultdict

# -------------------------- 配置 --------------------------
# 原始 PKL（带 Unnamed: 0 列，ID 为 id_xxx 格式）
kcat_km_pkl = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_with_log_feats.pkl"
# 三版 FASTA 目录（已生成）
v1_fasta_dir = "/mnt/usb3/code/wm/data/fasta_km/kcat_km_v1"

# 三版最终 PKL 输出路径
output_v1_pkl = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_v2_with_fold.pkl"
# ----------------------------------------------------------

def get_fold_mapping(fasta_dir):
    """
    功能：从 FASTA 目录中提取 {kcat_km的ID: fold编号} 映射
    参数：fasta_dir - 某版本的 FASTA 目录（如 kcat_km_v1）
    返回：fold_mapping - 字典，key=id_xxx，value=0~9（fold编号）
    """
    fold_mapping = {}
    for fold in range(10):
        fasta_path = os.path.join(fasta_dir, f"kcat_km_fold_{fold}.fasta")
        if not os.path.exists(fasta_path):
            print(f"警告：{fasta_path} 不存在，跳过该 fold")
            continue
        # 读取 FASTA，提取序列 ID 和对应的 fold
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = record.id  # 直接获取 kcat_km 的 ID（id_xxx 格式）
            fold_mapping[seq_id] = fold
    return fold_mapping

# 1. 读取原始 PKL 数据
print("读取原始 PKL 数据...")
kcat_km_df = pd.read_pickle(kcat_km_pkl)
print(f"原始 PKL 行数：{len(kcat_km_df)}")
print(f"ID 格式示例：{kcat_km_df['Unnamed: 0'].head().tolist()}")

# 确保 Unnamed: 0 列是字符串格式（与 FASTA 的 ID 完全一致）
kcat_km_df["Unnamed: 0"] = kcat_km_df["Unnamed: 0"].astype(str)

# 2. 生成 v1 最终 PKL（基础无偏+随机分配）
print("\n===== 生成 v1 最终 PKL =====")
v1_mapping = get_fold_mapping(v1_fasta_dir)
print(f"v1 映射关联 {len(v1_mapping)} 条序列（应等于总序列数）")

v1_df = kcat_km_df.copy()
# 删除原始 fold 列（避免冲突）
if "fold" in v1_df.columns:
    v1_df = v1_df.drop(columns=["fold"])
# 关联新 fold 列（用 Unnamed: 0 匹配 ID）
v1_df["fold"] = v1_df["Unnamed: 0"].map(v1_mapping).fillna(-1)

# 验证 fold 分布（确保无大量 -1，-1 表示未匹配到，理论上应为 0）
print(f"v1 fold 分布：\n{v1_df['fold'].value_counts().sort_index()}")
# 保存 v1 PKL
v1_df.to_pickle(output_v1_pkl)
print(f"v1 最终 PKL 保存至：{output_v1_pkl}")

print("\n✅ 三版最终数据集生成完成！")
print(f"使用建议：")
print(f"- 追求简单无偏：使用 v1（{output_v1_pkl}）")