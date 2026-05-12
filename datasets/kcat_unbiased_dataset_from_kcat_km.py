import pandas as pd
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os
from collections import defaultdict

# -------------------------- 配置（完整流程：Kcat/KM → Kcat，一步到位） --------------------------
# 基准数据集：已划分好的 Kcat/KM（含 fold 列）
kcat_km_pkl = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_with_log_feats.pkl"
# 目标数据集：待划分的 Kcat（修改为目标路径）
kcat_pkl = "/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete.pkl"

# 临时目录 & 输出目录
os.makedirs("temp", exist_ok=True)
kcat_km_fasta_dir = "/mnt/usb3/code/wm/data/fasta/kcat_km"  # Kcat/KM 分折目录（临时，不变）
kcat_out_dir = "/mnt/usb3/code/wm/data/fasta_kcat/kcat_v1_based_on_kcatkm"  # Kcat 分折输出目录（更新）
os.makedirs(kcat_km_fasta_dir, exist_ok=True)
os.makedirs(kcat_out_dir, exist_ok=True)

# 最终 PKL 输出路径（带 fold 列的 Kcat）（更新）
final_kcat_pkl = "/mnt/usb3/code/wm/data/kcat_data/kcat_v1_with_fold.pkl"

# CD-HIT 配置（和原始一致，不变）
similarity_threshold = 0.4
threads = 4

# 列名配置（若 Kcat 数据集列名不同，需在此修改，默认沿用原列名）
id_col = "Unnamed: 0"
seq_col = "Sequence"
fold_col = "fold"
# ----------------------------------------------------------

# ==================== 步骤 1：生成 Kcat 总 FASTA（比对对象 -i2）（全量更新为 Kcat） ====================
print("===== 步骤 1：生成 Kcat 联合总 FASTA =====")
kcat_df = pd.read_pickle(kcat_pkl)  # 读取 Kcat 数据集
kcat_df[id_col] = kcat_df[id_col].astype(str)
kcat_df[seq_col] = kcat_df[seq_col].astype(str).str.strip()

# 生成 Kcat 总 FASTA
kcat_all_fasta = "temp/kcat_all.fasta"  # 临时文件更新为 kcat 前缀
kcat_records = [
    SeqRecord(
        seq=str(row[seq_col]).strip(),
        id=str(row[id_col]),
        description=""
    )
    for _, row in kcat_df.iterrows()
]
SeqIO.write(kcat_records, kcat_all_fasta, "fasta")
print(f"Kcat 总 FASTA 保存至：{kcat_all_fasta}，含 {len(kcat_records)} 条序列")

# 构建 Kcat ID→序列映射（用于后续生成分折 FASTA）
kcat_id_to_seq = {}
for (_, row), rec in zip(kcat_df.iterrows(), kcat_records):
    seq_id = str(row[id_col])
    kcat_id_to_seq[seq_id] = rec
all_kcat_ids = [str(id_) for id_ in kcat_df[id_col].tolist()]

# ==================== 步骤 2：生成 Kcat/KM 分折 FASTA（比对基准 -i）（不变，仍为基准数据集） ====================
print("\n===== 步骤 2：生成 Kcat/KM 分折 FASTA（作为比对基准） =====")
kcat_km_df = pd.read_pickle(kcat_km_pkl)
print(f"Kcat/KM 数据集行数：{len(kcat_km_df)}")

# 验证 Kcat/KM 必要列
required_cols = [id_col, seq_col, fold_col]
for col in required_cols:
    if col not in kcat_km_df.columns:
        raise ValueError(f"Kcat/KM 数据集缺少必要列：{col}！")

# 格式预处理
kcat_km_df[id_col] = kcat_km_df[id_col].astype(str)
kcat_km_df[seq_col] = kcat_km_df[seq_col].astype(str)

# 生成 Kcat/KM 10 折 FASTA
for fold in range(10):
    fold_data = kcat_km_df[kcat_km_df[fold_col] == fold].copy()
    fold_records = [
        SeqRecord(
            seq=str(row[seq_col]).strip(),
            id=str(row[id_col]),
            description=""
        )
        for _, row in fold_data.iterrows()
    ]
    fold_fasta_path = os.path.join(kcat_km_fasta_dir, f"kcat_km_fold_{fold}.fasta")
    SeqIO.write(fold_records, fold_fasta_path, "fasta")
    print(f"Kcat/KM fold {fold}：{len(fold_records)} 条序列 → 保存至 {fold_fasta_path}")

print("✅ Kcat/KM 10 个分折 FASTA 生成完成！")

# ==================== 步骤 3：CD-HIT-2D 比对，标记相似 Kcat 序列（全量更新为 Kcat） ====================
print("\n===== 步骤 3：CD-HIT-2D 比对，标记与 Kcat/KM 高相似的 Kcat 序列 =====")
kcat_km_fold_similar = defaultdict(list)  # Kcat/KM fold → 相似 Kcat ID

for fold in range(10):
    fold_fasta = os.path.join(kcat_km_fasta_dir, f"kcat_km_fold_{fold}.fasta")
    if not os.path.exists(fold_fasta):
        print(f"警告：{fold_fasta} 不存在，跳过该 fold")
        continue

    # 运行 CD-HIT-2D（基准：Kcat/KM 分折，查询：Kcat 总 FASTA）
    output_prefix = f"temp/kcat_km_fold_{fold}_vs_kcat"  # 临时文件前缀更新为 vs_kcat
    os.system(
        f"cd-hit-2d -i {fold_fasta} -i2 {kcat_all_fasta} "  # 查询文件改为 Kcat 总 FASTA
        f"-o {output_prefix} -c {similarity_threshold} -n 2 -T {threads} -M 16000 -d 0"
    )

    # 解析 cluster 文件，提取相似 Kcat ID
    cluster_file = f"{output_prefix}.clstr"
    if os.path.exists(cluster_file):
        with open(cluster_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue
                seq_id = line.split(">")[-1].split("...")[0].strip()
                if seq_id in kcat_id_to_seq:  # 匹配 Kcat 的 ID 映射
                    kcat_km_fold_similar[fold].append(seq_id)

    # 去重（避免同一 Kcat 序列被多个 fold 标记）
    kcat_km_fold_similar[fold] = list(set(kcat_km_fold_similar[fold]))
    print(f"Kcat/KM fold {fold} 对应的高相似 Kcat 序列数：{len(kcat_km_fold_similar[fold])}")

# ==================== 步骤 4：给 Kcat 分配 fold（相似继承 + 随机分配）（全量更新为 Kcat） ====================
print("\n===== 步骤 4：给 Kcat 分配最终 fold =====")
kcat_fold_assign = {}  # Kcat ID → 最终 fold

# 第一步：相似序列继承 fold
for fold, similar_ids in kcat_km_fold_similar.items():
    for seq_id in similar_ids:
        if seq_id not in kcat_fold_assign:
            kcat_fold_assign[seq_id] = fold

print(f"已分配相似序列数：{len(kcat_fold_assign)}")

# 第二步：剩余序列随机分配（种子=42，确保可复现）
remaining_ids = [id_ for id_ in all_kcat_ids if id_ not in kcat_fold_assign]
print(f"剩余待分配序列数：{len(remaining_ids)}")

random.seed(42)
random.shuffle(remaining_ids)
for i, seq_id in enumerate(remaining_ids):
    kcat_fold_assign[seq_id] = i % 10  # 平均分配到 10 折

# ==================== 步骤 5：保存 Kcat 分折 FASTA（全量更新为 Kcat） ====================
print("\n===== 步骤 5：保存 Kcat 10 折 FASTA =====")
for fold in range(10):
    fold_ids = [id_ for id_, assign_fold in kcat_fold_assign.items() if assign_fold == fold]
    fold_records = [kcat_id_to_seq[id_] for id_ in fold_ids if id_ in kcat_id_to_seq]
    out_fasta_path = os.path.join(kcat_out_dir, f"kcat_fold_{fold}.fasta")  # 输出文件名更新为 kcat 前缀
    SeqIO.write(fold_records, out_fasta_path, "fasta")
    print(f"Kcat fold {fold}：{len(fold_records)} 条序列 → 保存至 {out_fasta_path}")

# ==================== 步骤 6：保存带 fold 列的 Kcat 最终 PKL（全量更新为 Kcat） ====================
print("\n===== 步骤 6：保存 Kcat 最终 PKL（含新 fold 列） =====")
kcat_final_df = kcat_df.copy()

# 删除 Kcat 原始 fold 列（如果存在），替换为新分配的 fold
if fold_col in kcat_final_df.columns:
    kcat_final_df = kcat_final_df.drop(columns=[fold_col])
kcat_final_df[fold_col] = kcat_final_df[id_col].map(kcat_fold_assign)

# 确保 fold 为整数类型
kcat_final_df[fold_col] = kcat_final_df[fold_col].astype(int)

# 保存 PKL（路径更新为 Kcat 目标路径）
kcat_final_df.to_pickle(final_kcat_pkl)
print(f"✅ Kcat 最终 PKL 保存至：{final_kcat_pkl}")

# 打印 fold 分布（验证结果）
print("\n📊 Kcat 最终 fold 分布：")
print(kcat_final_df[fold_col].value_counts().sort_index())

# ==================== 流程结束 ====================
print("\n" + "="*80)
print("✅ 完整流程结束！")
print("="*80)
print(f"输出文件汇总：")
print(f"1. Kcat 10 折 FASTA 目录：{kcat_out_dir}")
print(f"2. Kcat 带 fold 列 PKL：{final_kcat_pkl}")
print(f"3. 临时文件目录（可手动删除）：temp/")