import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm

# ====================== 配置参数 ======================
# 原始pkl文件路径
pkl_files = [
    "/mnt/usb1/wmx/eitlem/kcat_km_with_log_feats_processed.pkl",
    "/mnt/usb1/wmx/eitlem/kcat_v1_with_fold_processed.pkl",
    "/mnt/usb1/wmx/eitlem/km_v1_with_fold_processed.pkl"
]
# ESM特征提取相关配置
esm_extract_script = "./extract.py"  # ESM提取脚本路径（和你之前一致）
esm_model = "esm2_t33_650M_UR50D"  # 使用的ESM模型（和你之前一致）
fasta_file = "../Data/Feature/unique_seqs.fasta"  # 临时存储唯一序列的fasta文件
esm_output_dir = "../Data/Feature/esm2_t33_650M_UR50D_unique/"  # 唯一序列的ESM特征输出目录
seq_col = "Sequence"  # 序列列名
esm_id_col = "esm_id"  # 新增的ESM ID列名


# ====================== 步骤1：加载数据并提取所有唯一序列 ======================
def load_all_sequences(pkl_paths, seq_column):
    """加载所有pkl文件，提取Sequence列并去重"""
    all_seqs = []
    for pkl_path in pkl_paths:
        try:
            # 加载pkl文件（支持DataFrame或字典格式）
            data = pd.read_pickle(pkl_path) if os.path.exists(pkl_path) else None
            if data is None or seq_column not in data.columns:
                print(f"警告：{pkl_path} 文件不存在或无 {seq_column} 列，跳过")
                continue
            # 提取序列并去重（当前文件内）
            seqs = data[seq_column].dropna().unique().tolist()
            all_seqs.extend(seqs)
            print(f"从 {pkl_path} 提取到 {len(seqs)} 个不重复序列")
        except Exception as e:
            print(f"加载 {pkl_path} 失败：{e}")
            continue

    # 全局去重，得到唯一序列列表
    unique_seqs = list(set(all_seqs))
    # 为每个唯一序列分配ID（从0开始）
    seq_to_esm_id = {seq: idx for idx, seq in enumerate(unique_seqs)}
    print(f"\n总计提取到 {len(unique_seqs)} 个全局唯一序列")
    return unique_seqs, seq_to_esm_id


# 执行步骤1：获取唯一序列和序列-ID映射
unique_sequences, seq2id = load_all_sequences(pkl_files, seq_col)


# ====================== 步骤2：生成唯一序列的FASTA文件（供ESM提取特征） ======================
def write_fasta(sequences, seq_to_id, fasta_path):
    """将唯一序列写入FASTA文件，格式：>esm_id\n序列"""
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)  # 确保目录存在
    with open(fasta_path, 'w', encoding='utf-8') as f:
        for seq, esm_id in seq_to_id.items():
            f.write(f">{esm_id}\n{seq}\n")  # FASTA头部为esm_id，方便后续对应
    print(f"唯一序列已写入FASTA文件：{fasta_path}")


# 执行步骤2：生成FASTA文件
write_fasta(unique_sequences, seq2id, fasta_file)


# ====================== 步骤3：调用ESM提取唯一序列的特征 ======================
def extract_esm_features(extract_script, model, fasta_path, output_dir):
    """调用ESM脚本提取特征"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 构造ESM提取命令（和你之前的逻辑一致）
    cmd = (
        f"python {extract_script} {model} {fasta_path} {output_dir} "
        f"--repr_layers 33 --include per_tok"
    )
    print(f"\n执行ESM特征提取命令：{cmd}")
    # 执行命令
    exit_code = os.system(cmd)
    if exit_code == 0:
        print("ESM特征提取完成！")
    else:
        raise RuntimeError("ESM特征提取失败，请检查脚本/路径/模型是否正确")


# 执行步骤3：提取ESM特征
extract_esm_features(esm_extract_script, esm_model, fasta_file, esm_output_dir)


# ====================== 步骤4：精简ESM特征文件（只保留33层特征，和你之前的逻辑一致） ======================
def simplify_esm_features(output_dir, layer=33):
    """精简ESM特征文件，只保留指定层的特征"""
    file_list = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    print(f"\n开始精简 {len(file_list)} 个ESM特征文件...")
    for file_name in tqdm(file_list, desc="精简特征文件"):
        try:
            file_path = os.path.join(output_dir, file_name)
            data = torch.load(file_path)
            # 只保留指定层的特征
            data = data['representations'][layer]
            torch.save(data, file_path)
        except Exception as e:
            print(f"警告：精简 {file_name} 失败：{e}")
    print("ESM特征文件精简完成！")


# 执行步骤4：精简特征
simplify_esm_features(esm_output_dir, layer=33)


# ====================== 步骤5：为每个原始pkl文件添加esm_id列 ======================
def add_esm_id_to_pkl(pkl_paths, seq_column, esm_id_column, seq_to_id):
    """为每个pkl文件添加esm_id列"""
    for pkl_path in pkl_paths:
        try:
            # 加载原始数据
            data = pd.read_pickle(pkl_path)
            if seq_column not in data.columns:
                print(f"警告：{pkl_path} 无 {seq_column} 列，跳过添加esm_id")
                continue

            # 映射序列到esm_id（未匹配到的设为NaN）
            data[esm_id_column] = data[seq_column].map(seq_to_id)

            # 保存修改后的文件（覆盖原文件，也可改名为新文件，如加"_with_esm_id"后缀）
            # 如需保留原文件，可将下面的pkl_path改为 f"{pkl_path[:-4]}_with_esm_id.pkl"
            data.to_pickle(pkl_path)
            print(f"已为 {pkl_path} 添加 {esm_id_column} 列，保存完成")
        except Exception as e:
            print(f"处理 {pkl_path} 失败：{e}")


# 执行步骤5：添加esm_id列
add_esm_id_to_pkl(pkl_files, seq_col, esm_id_col, seq2id)

print("\n所有步骤执行完成！")