import pandas as pd
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import os
from collections import defaultdict

# -------------------------- 配置（关键修正：明确 KM 相关路径） --------------------------
# Kcat/KM 联合特征 PKL（待划分数据）
kcat_km_pkl = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_with_log_feats.pkl"
# KM 数据集 PKL（含 fold 列，用于生成分折 FASTA）
km_pkl = "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats.pkl"
# KM 分折 FASTA 输出目录（临时生成，作为 CD-HIT 比对基准）
km_fasta_dir = "/mnt/usb3/code/wm/data/fasta/km"
# Kcat/KM 联合总 FASTA（临时文件）
kcat_km_all_fasta = "temp/kcat_km_all.fasta"
# 最终 Kcat/KM 分折 FASTA 输出目录（依据 KM 划分的 V1 版本）
v1_out_dir = "/mnt/usb3/code/wm/data/fasta_km/kcat_km_v1"

# CD-HIT 配置
similarity_threshold = 0.4  # 相似性阈值（和原 V1 一致）
threads = 4  # 线程数
# ----------------------------------------------------------

# 创建所需目录（含临时目录和输出目录）
os.makedirs(km_fasta_dir, exist_ok=True)  # KM 分折 FASTA 目录
os.makedirs(v1_out_dir, exist_ok=True)    # 最终输出目录
os.makedirs("temp", exist_ok=True)        # 临时文件目录

# -------------------------- 步骤 1：读取 Kcat/KM 联合数据，生成总 FASTA --------------------------
print("===== 步骤 1：生成 Kcat/KM 联合总 FASTA =====")
kcat_km_df = pd.read_pickle(kcat_km_pkl)
print(f"Kcat/KM 联合数据行数：{len(kcat_km_df)}")
print(f"ID 列示例：{kcat_km_df['Unnamed: 0'].head().tolist()}")
print(f"序列列示例：{kcat_km_df['Sequence'].head().tolist()}")

# 生成联合总 FASTA 记录（ID=Unnamed: 0，序列=Sequence）
records = [
    SeqRecord(
        seq=str(row["Sequence"]).strip(),
        id=str(row["Unnamed: 0"]),  # 确保 ID 为字符串，和 KM 一致
        description=""
    )
    for _, row in kcat_km_df.iterrows()
]

# 保存联合总 FASTA
SeqIO.write(records, kcat_km_all_fasta, "fasta")
print(f"联合总 FASTA 保存至：{kcat_km_all_fasta}，含 {len(records)} 条序列")

# -------------------------- 修复：生成 Kcat/KM ID→序列映射 --------------------------
# 错误原因：iterrows() 返回 (索引, 行数据) 两个值，不能用 3 个变量解包
# 修复方式：用 enumerate 遍历 records，或直接 zip 后解包两个值
kcat_km_id_to_seq = {}
for (_, row), rec in zip(kcat_km_df.iterrows(), records):
    seq_id = str(row["Unnamed: 0"])
    kcat_km_id_to_seq[seq_id] = rec
# ------------------------------------------------------------------------------------------

all_kcat_km_ids = [str(id_) for id_ in kcat_km_df["Unnamed: 0"].tolist()]

# -------------------------- 步骤 2：从 KM 数据集生成 10 个分折 FASTA（关键新增步骤） --------------------------
print("\n===== 步骤 2：生成 KM 分折 FASTA（作为比对基准） =====")
# 读取 KM 数据集
km_df = pd.read_pickle(km_pkl)
print(f"KM 数据集行数：{len(km_df)}")
print(f"KM 数据集列：{km_df.columns.tolist()}")

# 验证 KM 数据集必要列（确保有 ID、序列、fold）
required_km_cols = ["Unnamed: 0", "Sequence", "fold"]
for col in required_km_cols:
    if col not in km_df.columns:
        raise ValueError(f"KM 数据集缺少必要列：{col}！请根据实际列名修改配置")

# 确保 KM 的 ID 和序列为字符串格式（避免匹配失败）
km_df["Unnamed: 0"] = km_df["Unnamed: 0"].astype(str)
km_df["Sequence"] = km_df["Sequence"].astype(str)

# 按 KM 自带的 fold 列，生成 10 个分折 FASTA
for fold in range(10):
    # 筛选当前 fold 的 KM 数据
    fold_km_data = km_df[km_df["fold"] == fold].copy()
    # 生成 FASTA 记录
    fold_records = [
        SeqRecord(
            seq=str(row["Sequence"]).strip(),
            id=str(row["Unnamed: 0"]),
            description=""
        )
        for _, row in fold_km_data.iterrows()
    ]
    # 保存 KM 分折 FASTA
    km_fold_fasta_path = os.path.join(km_fasta_dir, f"km_fold_{fold}.fasta")
    SeqIO.write(fold_records, km_fold_fasta_path, "fasta")
    print(f"KM fold {fold}：{len(fold_records)} 条序列 → 保存至 {km_fold_fasta_path}")

print("✅ KM 10 个分折 FASTA 生成完成！")

# -------------------------- 步骤 3：CD-HIT 标记与 KM 高相似的 Kcat/KM 序列 --------------------------
print("\n===== 步骤 3：标记与 KM 高相似的序列 =====")
# 存储「KM fold → 相似的 Kcat/KM 序列 ID」映射
km_fold_similar = defaultdict(list)

for fold in range(10):
    # 读取当前 fold 的 KM 分折 FASTA（步骤 2 生成的）
    km_fold_fasta = os.path.join(km_fasta_dir, f"km_fold_{fold}.fasta")
    if not os.path.exists(km_fold_fasta):
        print(f"警告：{km_fold_fasta} 不存在，跳过该 fold")
        continue

    # 运行 CD-HIT-2D：比对 KM 分折 FASTA 和 Kcat/KM 联合 FASTA
    output_prefix = f"temp/km_fold_{fold}_vs_kcat_km"
    os.system(
        f"cd-hit-2d -i {km_fold_fasta} -i2 {kcat_km_all_fasta} "
        f"-o {output_prefix} -c {similarity_threshold} -n 2 -T {threads} -M 16000 -d 0"
    )

    # 解析 CD-HIT 输出的 cluster 文件，提取相似序列 ID
    cluster_file = f"{output_prefix}.clstr"
    if os.path.exists(cluster_file):
        with open(cluster_file, "r") as f:
            for line in f:
                if line.startswith(">"):  # 跳过 cluster 标题行
                    continue
                # 提取序列 ID（格式：>id_xxx... → 取 id_xxx 部分）
                seq_id = line.split(">")[-1].split("...")[0].strip()
                # 只保留 Kcat/KM 联合数据中存在的 ID
                if seq_id in kcat_km_id_to_seq:
                    km_fold_similar[fold].append(seq_id)

    # 去重（避免同一序列被多次标记）
    km_fold_similar[fold] = list(set(km_fold_similar[fold]))
    print(f"KM fold {fold} 对应的高相似 Kcat/KM 序列数：{len(km_fold_similar[fold])}")

# -------------------------- 步骤 4：生成 V1 版本（依据 KM 划分，基础无偏+随机分配） --------------------------
print("\n===== 步骤 4：生成 V1 划分（依据 KM 相似性+随机分配） =====")
v1_fold_assign = {}  # 存储最终的「Kcat/KM ID → fold」映射

# 第一步：分配与 KM 高相似的序列（跟随对应 KM 的 fold）
for fold, similar_ids in km_fold_similar.items():
    for seq_id in similar_ids:
        if seq_id not in v1_fold_assign:  # 避免重复分配
            v1_fold_assign[seq_id] = fold

print(f"已分配相似序列数：{len(v1_fold_assign)}")

# 第二步：剩余未分配序列，随机分配到 10 个 fold（保持无偏）
remaining_ids = [id_ for id_ in all_kcat_km_ids if id_ not in v1_fold_assign]
print(f"剩余待分配序列数：{len(remaining_ids)}")

# 随机打乱（固定种子确保可复现，和原 V1 一致）
random.seed(42)
random.shuffle(remaining_ids)

# 平均分配到 10 个 fold
for i, seq_id in enumerate(remaining_ids):
    v1_fold_assign[seq_id] = i % 10  # 按索引取模分配

# 第三步：保存 V1 分折 FASTA（最终结果）
for fold in range(10):
    # 筛选当前 fold 的所有 Kcat/KM ID
    fold_ids = [id_ for id_, assign_fold in v1_fold_assign.items() if assign_fold == fold]
    # 对应的序列记录
    fold_records = [kcat_km_id_to_seq[id_] for id_ in fold_ids if id_ in kcat_km_id_to_seq]
    # 保存 FASTA
    out_fasta_path = os.path.join(v1_out_dir, f"kcat_km_fold_{fold}.fasta")
    SeqIO.write(fold_records, out_fasta_path, "fasta")
    print(f"V1 fold {fold}：{len(fold_records)} 条序列 → 保存至 {out_fasta_path}")

# -------------------------- 结果验证与提示 --------------------------
print("\n" + "="*80)
print("✅ 依据 KM 划分的 Kcat/KM V1 数据集生成完成！")
print("="*80)
print(f"核心输出目录：{v1_out_dir}")
print(f"包含文件：kcat_km_fold_0.fasta ~ kcat_km_fold_9.fasta（共 10 个分折）")
print(f"划分逻辑：")
print(f"1. 先将与 KM fold X 高相似的 Kcat/KM 序列分配到 fold X；")
print(f"2. 剩余序列随机分配（种子=42，确保可复现）；")
print(f"3. 保持基础无偏特性，与原 V1 逻辑一致。")
print(f"\n后续使用：可直接用你之前的代码，读取这些分折 FASTA 给 Kcat/KM 数据贴 fold 标签。")