import os
import re
import gc
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from joblib import load

# ------------------------- 1. 基础设置 -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

pkl_path = "/mnt/usb3/code/wm/data/km_data/km_with_all_feats.pkl"
output_path = "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats.pkl"
batch_size = 32
max_len = 1000

sequence_col = "Sequence"
target_feat_col = "new_ezy_feat"
seq_len_col = "seq_length"
original_ezy_col = "ezy_feat"


# ------------------------- 2. 安全读写工具 -------------------------
def safe_pickle_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"加载pickle文件失败：{e}")


# ------------------------- 3. ProtT5模型加载 -------------------------
def load_prot5_model(model_path="/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50"):
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    print(f"✅ ProtT5模型加载完成（设备：{device}）")
    return tokenizer, model


# ------------------------- 4. 核心修改：修复is_feat_missing函数（兼容数组） -------------------------
def fill_missing_feat_by_unique_seq(df, tokenizer, model):
    # 步骤1：统计new_ezy_feat空缺情况（核心修改：先判断元素类型）
    print("\n" + "=" * 50)
    print(f"1. 统计 {target_feat_col} 列的空缺情况")
    print("=" * 50)

    def is_feat_missing(val):
        """判断特征是否空缺：分数组和非数组两种情况"""
        # 情况1：元素是数组/列表类型
        if isinstance(val, (list, np.ndarray)):
            # 空数组视为空缺
            return len(val) == 0
        # 情况2：元素是非数组类型（如NaN、None、空字符串）
        else:
            return pd.isna(val) or str(val).strip() == ""

    # 统计全量数据空缺（用apply避免逐元素循环的歧义）
    # 先生成每行是否空缺的布尔 Series，再统计
    df["is_feat_missing"] = df[target_feat_col].apply(is_feat_missing)
    total_rows = len(df)
    missing_feat_cnt = df["is_feat_missing"].sum()
    valid_feat_cnt = total_rows - missing_feat_cnt
    missing_rate = missing_feat_cnt / total_rows if total_rows > 0 else 0

    print(f"全量数据集总行数：{total_rows} 条")
    print(f"{target_feat_col} 有效行数：{valid_feat_cnt} 条（{valid_feat_cnt / total_rows:.2%}）")
    print(f"{target_feat_col} 空缺行数：{missing_feat_cnt} 条（{missing_rate:.2%}）")

    # 步骤2：提取“空缺特征对应的唯一序列”
    print("\n" + "=" * 50)
    print(f"2. 提取 {target_feat_col} 空缺对应的唯一序列")
    print("=" * 50)

    # 筛选特征空缺的样本（用之前生成的布尔列，避免重复计算）
    df_missing_feat = df[df["is_feat_missing"]].copy()
    # 提取唯一序列（需补特征的序列）
    missing_sequences = df_missing_feat[sequence_col].tolist()
    unique_missing_seqs = list(set(missing_sequences))
    total_unique_missing = len(unique_missing_seqs)

    print(f"特征空缺的样本数：{len(df_missing_feat)} 条")
    print(f"特征空缺样本中的唯一序列数：{total_unique_missing} 条（仅需处理这部分）")
    print(f"重复的空缺序列数：{len(missing_sequences) - total_unique_missing} 条（后续复用特征）")

    # 无需要补的序列，直接返回
    if total_unique_missing == 0:
        print(f"\n⚠️ 无需要补特征的序列，删除临时列后返回")
        df = df.drop("is_feat_missing", axis=1)
        return df

    # 步骤3：预处理需补特征的唯一序列
    print(f"\n⏳ 预处理 {total_unique_missing} 条需补特征的唯一序列...")
    seq_to_original_len = {seq: len(seq) for seq in unique_missing_seqs}
    formatted_seqs = []
    for seq in unique_missing_seqs:
        truncated = seq[:500] + seq[-500:] if len(seq) > max_len else seq
        formatted = ' '.join(list(truncated))
        formatted = re.sub(r"[UZOB]", "X", formatted)
        formatted_seqs.append(formatted)
    print(f"✅ 需补特征序列预处理完成")

    # 步骤4：批量提取特征
    print("\n" + "=" * 50)
    print(f"3. 批量提取 {total_unique_missing} 条序列的特征")
    print("=" * 50)
    seq_to_new_feat = dict()

    for start in tqdm(range(0, total_unique_missing, batch_size), desc="补特征进度"):
        end = min(start + batch_size, total_unique_missing)
        batch_raw_seqs = unique_missing_seqs[start:end]
        batch_formatted = formatted_seqs[start:end]

        # Tokenize
        inputs = tokenizer(
            batch_formatted,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_len + 2,
            return_tensors="pt"
        ).to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        # 提取有效特征
        for i in range(len(batch_raw_seqs)):
            seq = batch_raw_seqs[i]
            valid_len = (inputs["attention_mask"][i] == 1).sum().item() - 2
            residue_feat = last_hidden_state[i, 1:1 + valid_len, :].cpu().numpy()
            seq_to_new_feat[seq] = residue_feat

        # 清理内存
        del inputs, outputs, last_hidden_state, batch_raw_seqs, batch_formatted
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✅ 需补特征序列处理完成（共生成 {len(seq_to_new_feat)} 条新特征）")

    # 步骤5：合并回全量数据集
    print("\n" + "=" * 50)
    print(f"4. 将补好的特征合并回全量数据集")
    print("=" * 50)

    def fill_feat(row):
        seq = row[sequence_col]
        # 仅对“原特征空缺”且“有新特征”的样本填充
        if row["is_feat_missing"] and seq in seq_to_new_feat:
            row[target_feat_col] = seq_to_new_feat[seq]
            row[seq_len_col] = seq_to_original_len[seq]
        return row

    df_filled = df.apply(fill_feat, axis=1)

    # 步骤6：验证填充结果（删除临时列）
    df_filled = df_filled.drop("is_feat_missing", axis=1)
    # 重新统计填充后的空缺情况
    df_filled["post_is_missing"] = df_filled[target_feat_col].apply(is_feat_missing)
    post_missing_cnt = df_filled["post_is_missing"].sum()
    post_valid_cnt = total_rows - post_missing_cnt
    filled_cnt = valid_feat_cnt - post_valid_cnt  # 本次填充的数量（注意符号：post_valid > valid）

    print(f"填充后 {target_feat_col} 有效行数：{post_valid_cnt} 条（{post_valid_cnt / total_rows:.2%}）")
    print(f"填充后 {target_feat_col} 剩余空缺行数：{post_missing_cnt} 条（{post_missing_cnt / total_rows:.2%}）")
    print(f"本次共填充：{abs(filled_cnt)} 条空缺")  # 用abs避免负号（因post_valid > valid）
    df_filled = df_filled.drop("post_is_missing", axis=1)

    return df_filled


# ------------------------- 5. 主流程 -------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行：new_ezy_feat空缺检查→唯一序列补特征→合并全量数据")
    print("=" * 50)

    # 加载全量数据
    print(f"\n1. 加载全量数据集：{pkl_path}")
    df = safe_pickle_load(pkl_path)
    necessary_cols = [sequence_col, target_feat_col, original_ezy_col, seq_len_col]
    assert all(col in df.columns for col in
               necessary_cols), f"缺少必要列：{[col for col in necessary_cols if col not in df.columns]}"
    print(f"✅ 全量数据加载完成：{len(df)} 行 × {len(df.columns)} 列")

    # 加载模型
    tokenizer, model = load_prot5_model()

    # 补特征并合并
    df_filled = fill_missing_feat_by_unique_seq(df, tokenizer, model)

    # 保存结果
    df_filled.to_pickle(output_path)
    print(f"\n" + "=" * 50)
    print("✅ 全部流程完成！")
    print("=" * 50)
    print(f"📁 补全后的全量数据集保存路径：{output_path}")
    print(f"📊 最终数据规模：{len(df_filled)} 行 × {len(df_filled.columns)} 列")