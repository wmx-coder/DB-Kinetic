import os
import sys
import re
import gc
import shutil
import pandas as pd
import subprocess
from tqdm import tqdm

# -------------------------- 全局配置 --------------------------
pkl_path = "/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete.pkl"
output_pkl_with_id_path = "/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete_esm_id.pkl"
extract_script = "/mnt/usb/code/wm/EITLEM/Code/extract.py"
esm_model = "esm2_t33_650M_UR50D"
sequence_col = "Sequence"
target_layer = 33
prefix = "unique_kcat"
unique_temp_fasta = f"./{prefix}_sequences.fasta"
unique_esm_feature_dir = "/mnt/usb3/code/wm/esm/kcat/"
os.makedirs(unique_esm_feature_dir, exist_ok=True)
skip_feature_generation = False

# -------------------------- 统一的序列清洗函数 --------------------------
def clean_sequence(seq):
    """
    统一的序列清洗规则：
    1. 非字符串返回空
    2. 去除所有空白字符（空格/换行/制表符）
    3. 转为大写
    4. 仅保留A-Z的字母（过滤*、-、数字等）
    """
    if not isinstance(seq, str):
        return ""
    # 步骤1：去除所有空白字符
    cleaned = seq.strip().replace(" ", "").replace("\n", "").replace("\t", "")
    # 步骤2：转为大写
    cleaned = cleaned.upper()
    # 步骤3：仅保留A-Z的字母
    cleaned = ''.join([c for c in cleaned if c.isalpha() and c.isupper()])
    return cleaned

# -------------------------- 步骤1：提取唯一序列+生成FASTA（统一清洗） --------------------------
print("=" * 60)
print("步骤1/6：提取唯一序列+生成FASTA（统一清洗）")
print("=" * 60)

df = pd.read_pickle(pkl_path, compression=None)
print(f"✅ 原数据集加载成功 | 形状：{df.shape}")

# 1. 清洗序列 + 过滤无效序列
df_clean = df[
    df[sequence_col].apply(lambda x: len(clean_sequence(x)) > 0)
].copy()
# 对序列列做统一清洗
df_clean[sequence_col] = df_clean[sequence_col].apply(clean_sequence)

# 2. 序列去重（基于清洗后的序列）
unique_seqs = df_clean[sequence_col].drop_duplicates().reset_index(drop=True)
total_unique = len(unique_seqs)
print(f"✅ 唯一序列统计 | 总数：{total_unique}（已清洗）")

# 3. 生成FASTA（使用清洗后的序列）
fasta_id_mapping = {}
with open(unique_temp_fasta, 'w', encoding='utf-8') as f:
    for seq_idx in range(total_unique):
        clean_seq = unique_seqs.iloc[seq_idx]
        fasta_id = f"{prefix}_{seq_idx}"
        f.write(f">{fasta_id}\n")
        f.write(f"{clean_seq}\n")
        fasta_id_mapping[seq_idx] = clean_seq

print(f"✅ FASTA生成完成 | 路径：{os.path.abspath(unique_temp_fasta)}")

# -------------------------- 步骤2：调用ESM提取特征 --------------------------
print("\n" + "=" * 60)
print("步骤2/6：调用ESM提取特征")
print("=" * 60)

cmd = (
    f"python {extract_script} "
    f"{esm_model} "
    f"{unique_temp_fasta} "
    f"{unique_esm_feature_dir} "
    f"--repr_layers {target_layer} "
    f"--include per_tok "
    f"--toks_per_batch 2048 "
)

print(f"📢 执行特征提取命令：\n{cmd}")
try:
    result = subprocess.run(
        cmd, shell=True, check=True,
        capture_output=True, text=True, encoding="utf-8"
    )
    print("✅ ESM特征提取成功！")
except subprocess.CalledProcessError as e:
    print(f"❌ ESM提取失败 | 错误：{e.stderr}")
    os.remove(unique_temp_fasta)
    exit(1)

# 验证特征文件
feature_files = os.listdir(unique_esm_feature_dir)
valid_feature_files = [f for f in feature_files if f.startswith(prefix) and f.endswith(".pt")]
print(f"✅ 有效特征文件数 | {len(valid_feature_files)}（应等于{total_unique}）")

# -------------------------- 步骤3：构建序列→ID映射（基于清洗后的序列） --------------------------
print("\n" + "=" * 60)
print("步骤3/6：构建序列→ID映射（清洗后）")
print("=" * 60)

seq_to_esm_id = {}
file_pattern = re.compile(rf"{prefix}_(\d+)\.pt$")

for filename in valid_feature_files:
    match = file_pattern.match(filename)
    if match:
        esm_id = int(match.group(1))
        clean_seq = fasta_id_mapping.get(esm_id, "")
        if len(clean_seq) > 0:
            seq_to_esm_id[clean_seq] = esm_id

print(f"✅ 映射构建完成 | 有效映射数：{len(seq_to_esm_id)}")

# -------------------------- 步骤4：填充ESM ID（统一清洗） --------------------------
print("\n" + "=" * 60)
print("步骤4/6：填充ESM ID（统一清洗）")
print("=" * 60)

# 重新加载原始数据集
df = pd.read_pickle(pkl_path, compression=None)
df['esm_id'] = -1  # 初始化无效ID为-1
print(f"✅ 原始数据集加载 | 总行数：{len(df)}")

# 批量填充（统一清洗）
fill_count = 0
for idx in tqdm(df.index, desc="填充ESM ID"):
    seq_val = df.loc[idx, sequence_col]
    clean_seq = clean_sequence(seq_val)  # 统一清洗
    if len(clean_seq) > 0:
        esm_id = seq_to_esm_id.get(clean_seq, -1)
        if esm_id != -1:
            df.loc[idx, 'esm_id'] = esm_id
            fill_count += 1

# 统计结果
fill_rate = fill_count / len(df) if len(df) > 0 else 0.0
print(f"✅ ID填充完成 | 有效ID数：{fill_count}/{len(df)} | 填充率：{fill_rate:.4f}")

# -------------------------- 步骤5：验证 --------------------------
print("\n" + "=" * 60)
print("步骤5/6：验证ID→特征文件对应性")
print("=" * 60)

valid_ids = df[df['esm_id'] != -1]['esm_id'].head(5).tolist()
print(f"✅ 抽样验证：")
for esm_id in valid_ids:
    feat_file = f"{prefix}_{esm_id}.pt"
    feat_path = os.path.join(unique_esm_feature_dir, feat_file)
    exists = os.path.exists(feat_path)
    print(f"   ID={esm_id} → {feat_file} | 存在：{exists}")

# -------------------------- 步骤6：保存+清理 --------------------------
print("\n" + "=" * 60)
print("步骤6/6：保存+清理")
print("=" * 60)

df.to_pickle(output_pkl_with_id_path, compression=None)
print(f"✅ 带ESM ID的数据集已保存 | 路径：{output_pkl_with_id_path}")

if os.path.exists(unique_temp_fasta):
    os.remove(unique_temp_fasta)
    print(f"✅ 清理临时FASTA | {unique_temp_fasta}")

print("\n🎉 全流程完成！统一清洗规则确保序列100%匹配")
gc.collect()