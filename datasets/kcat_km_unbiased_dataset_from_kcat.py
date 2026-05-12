import pandas as pd
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os
from collections import defaultdict

# -------------------------- 配置 --------------------------
kcat_km_pkl = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_with_log_feats.pkl"
kcat_fasta_dir = "/mnt/usb3/code/wm/data/fasta/kcat"
kcat_km_all_fasta = "temp/kcat_km_all.fasta"
# 三版输出目录
v1_out_dir = "/mnt/usb3/code/wm/data/fasta/kcat_km_v1"  # 基础无偏（随机分配）
v2_out_dir = "/mnt/usb3/code/wm/data/fasta/kcat_km_v2"  # 严格无偏（剔除相似）
v3_out_dir = "/mnt/usb3/code/wm/data/fasta/kcat_km_v3"  # 基础无偏（尽量平均）
similarity_threshold = 0.4
threads = 4
# ----------------------------------------------------------

# 创建三版输出目录
os.makedirs(v1_out_dir, exist_ok=True)
os.makedirs(v2_out_dir, exist_ok=True)
os.makedirs(v3_out_dir, exist_ok=True)
os.makedirs("temp1", exist_ok=True)

# 1. 读取 PKL，生成 kcat_km 总 FASTA（id_xxx 作为 ID）
print("读取 PKL 数据，生成 kcat_km 总 FASTA...")
kcat_km_df = pd.read_pickle(kcat_km_pkl)
print(f"PKL 行数：{len(kcat_km_df)}，Unnamed: 0 列类型：{kcat_km_df['Unnamed: 0'].dtype}")
print(f"前 5 个 ID 示例：{kcat_km_df['Unnamed: 0'].head().tolist()}")

# 生成 FASTA 记录
records = [
    SeqRecord(
        seq=str(row["Sequence"]).strip(),
        id=row["Unnamed: 0"],
        description=""
    )
    for _, row in kcat_km_df.iterrows()
]

# 保存总 FASTA
SeqIO.write(records, kcat_km_all_fasta, "fasta")
print(f"总 FASTA 生成完成：{kcat_km_all_fasta}，含 {len(records)} 条序列")

# 2. 构建 ID→序列映射
km_id_to_seq = {}
for (_, row), rec in zip(kcat_km_df.iterrows(), records):
    seq_id = row["Unnamed: 0"]
    km_id_to_seq[seq_id] = rec
all_km_ids = kcat_km_df["Unnamed: 0"].tolist()

# 3. CD-HIT 标记与 kcat 高相似的序列（三版共用此结果）
print("\n===== 标记与 kcat 高相似的序列 =====")
kcat_fold_similar = defaultdict(list)

for fold in range(10):
    kcat_fasta = os.path.join(kcat_fasta_dir, f"kcat_fold_{fold}.fasta")
    if not os.path.exists(kcat_fasta):
        print(f"警告：{kcat_fasta} 不存在，跳过")
        continue

    # 运行 CD-HIT-2D
    output_prefix = f"temp/kcat_fold_{fold}_vs_km"
    os.system(
        f"cd-hit-2d -i {kcat_fasta} -i2 {kcat_km_all_fasta} "
        f"-o {output_prefix} -c {similarity_threshold} -n 2 -T {threads} -M 16000 -d 0"
    )

    # 解析 cluster 文件
    cluster_file = f"{output_prefix}.clstr"
    if os.path.exists(cluster_file):
        with open(cluster_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue
                seq_id = line.split(">")[-1].split("...")[0].strip()
                if seq_id in km_id_to_seq:
                    kcat_fold_similar[fold].append(seq_id)

    # 去重
    kcat_fold_similar[fold] = list(set(kcat_fold_similar[fold]))
    print(f"kcat fold {fold}：{len(kcat_fold_similar[fold])} 条高相似序列")

# 4. v1：基础无偏（随机分配，原逻辑不变）
print("\n===== 生成 v1 划分（基础无偏+随机分配） =====")
v1_fold_assign = {}
# 分配高相似序列
for fold, seq_ids in kcat_fold_similar.items():
    for seq_id in seq_ids:
        if seq_id not in v1_fold_assign:
            v1_fold_assign[seq_id] = fold
# 剩余序列随机分配
remaining_ids_v1 = [id for id in all_km_ids if id not in v1_fold_assign]
random.seed(42)
random.shuffle(remaining_ids_v1)
for i, seq_id in enumerate(remaining_ids_v1):
    v1_fold_assign[seq_id] = i % 10
# 保存 v1 FASTA
for fold in range(10):
    fold_ids = [id for id, f in v1_fold_assign.items() if f == fold]
    fold_records = [km_id_to_seq[id] for id in fold_ids if id in km_id_to_seq]
    out_path = os.path.join(v1_out_dir, f"kcat_km_fold_{fold}.fasta")
    SeqIO.write(fold_records, out_path, "fasta")
    print(f"v1 fold {fold}：{len(fold_records)} 条序列")

# 5. v2：严格无偏（剔除相似+随机分配，原逻辑不变）
print("\n===== 生成 v2 划分（严格无偏+随机分配） =====")
conflict_ids = set()
for seq_ids in kcat_fold_similar.values():
    conflict_ids.update(seq_ids)
print(f"冲突序列总数：{len(conflict_ids)}")
clean_ids = [id for id in all_km_ids if id not in conflict_ids]
print(f"干净序列数：{len(clean_ids)}")
# 随机分配
random.seed(42)
random.shuffle(clean_ids)
v2_fold_assign = {id: i % 10 for i, id in enumerate(clean_ids)}
# 保存 v2 FASTA
for fold in range(10):
    fold_ids = [id for id, f in v2_fold_assign.items() if f == fold]
    fold_records = [km_id_to_seq[id] for id in fold_ids if id in km_id_to_seq]
    out_path = os.path.join(v2_out_dir, f"kcat_km_fold_{fold}.fasta")
    SeqIO.write(fold_records, out_path, "fasta")
    print(f"v2 fold {fold}：{len(fold_records)} 条序列")

# 6. v3：基础无偏（尽量平均，不强制差异）【核心优化】
print("\n===== 生成 v3 划分（基础无偏+尽量平均） =====")
v3_fold_assign = {}
# 第一步：先分配高相似序列（和 v1 一致，尊重相似性逻辑）
for fold, seq_ids in kcat_fold_similar.items():
    for seq_id in seq_ids:
        if seq_id not in v3_fold_assign:
            v3_fold_assign[seq_id] = fold

# 第二步：统计已分配序列的 fold 分布，计算平均水平
fold_count = defaultdict(int)
for fold in v3_fold_assign.values():
    fold_count[fold] += 1
total_assigned = len(v3_fold_assign)
total_remaining = len(all_km_ids) - total_assigned
avg_per_fold = (len(all_km_ids) / 10)  # 理论平均水平（不强制整数）
print(f"理论平均每条 fold：{avg_per_fold:.1f} 条")

# 第三步：剩余序列分配策略——优先分给当前数量低于平均的 fold
remaining_ids_v3 = [id for id in all_km_ids if id not in v3_fold_assign]
random.seed(42)
random.shuffle(remaining_ids_v3)

for seq_id in remaining_ids_v3:
    # 找到当前数量 < 平均水平的 fold（按 0~9 顺序，避免极端不均）
    target_fold = None
    for fold in range(10):
        current_count = fold_count.get(fold, 0)
        if current_count < avg_per_fold:
            target_fold = fold
            break
    # 若所有 fold 都≥平均，就按顺序分配（避免某一个 fold 一直空着）
    if target_fold is None:
        target_fold = min(fold_count.keys(), key=lambda x: fold_count[x])

    # 分配序列并更新计数
    v3_fold_assign[seq_id] = target_fold
    fold_count[target_fold] = fold_count.get(target_fold, 0) + 1

# 保存 v3 FASTA
for fold in range(10):
    fold_ids = [id for id, f in v3_fold_assign.items() if f == fold]
    fold_records = [km_id_to_seq[id] for id in fold_ids if id in km_id_to_seq]
    out_path = os.path.join(v3_out_dir, f"kcat_km_fold_{fold}.fasta")
    SeqIO.write(fold_records, out_path, "fasta")
    print(f"v3 fold {fold}：{len(fold_records)} 条序列")

#打印 v3 最终分布统计（看是否避免极端不均）
v3_dist = [len([id for id, f in v3_fold_assign.items() if f == fold]) for fold in range(10)]
print(f"\nv3 各 fold 数据量范围：{min(v3_dist)} ~ {max(v3_dist)}（差异：{max(v3_dist) - min(v3_dist)}）")
print("\n✅ 三版 FASTA 划分完成！")
print(f"- v1：{v1_out_dir}（基础无偏+随机分配）")
print(f"- v2：{v2_out_dir}（严格无偏+随机分配）")
print(f"- v3：{v3_out_dir}（基础无偏+尽量平均）")