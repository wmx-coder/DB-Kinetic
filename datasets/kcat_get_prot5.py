import datetime
import os
import re
import gc
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from joblib import dump, load

# ------------------------- 1. 基础设置 -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

pkl_path = "/mnt/usb/code/wm/catapro/datasets/kcat_data/kcat-data_feats.pkl"
output_path = "/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_with_new_ezy.pkl"
breakpoint_path = "/mnt/usb/code/wm/catapro/datasets/kcat_data/ezy_batch_breakpoint_unique.pkl"
batch_size = 32  # 保持你当前的批次大小
max_len = 1000

original_ezy_col = "ezy_feat"
sequence_col = "Sequence"

# ------------------------- 2. 安全读写工具 -------------------------
# 【修改1】删除safe_joblib_dump（不再保存断点，无需此函数）
def safe_pickle_load(path):
    if not os.path.exists(path):
        return None
    try:
        return load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

# ------------------------- 3. 断点管理函数 -------------------------
def load_batch_breakpoint():
    folder = os.path.dirname(breakpoint_path)
    base = os.path.basename(breakpoint_path)
    backup_files = [f for f in os.listdir(folder) if f.startswith(base + ".backup_")]
    backup_files.sort(reverse=True)  # 按时间戳倒序，优先加载最新备份

    for candidate in backup_files + [base]:
        path = os.path.join(folder, candidate)
        data = safe_pickle_load(path)
        if data and "completed_count" in data and "seq_to_feat" in data:
            print(f"✅ 成功加载断点：{path}（已完成 {data['completed_count']} 条唯一序列）")
            last_index = data.get("last_index", data["completed_count"])
            # 【修改2】仅返回已处理的特征、长度和进度，忽略unique_sequences（用新生成的去重序列）
            return data["seq_to_feat"], data["seq_to_len"], last_index

    print("⚠️ 未找到有效断点文件，从头开始")
    return dict(), dict(), 0  # 无断点时返回空字典和0进度

# 【修改3】删除save_batch_breakpoint函数（彻底不保存新断点）

# ------------------------- 4. ProtT5模型加载（保持原样，不修改） -------------------------
def load_prot5_model(model_path="/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50"):
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    return tokenizer, model

# ------------------------- 5. 唯一序列特征提取 -------------------------
def unique_seq_extract(original_sequences, tokenizer, model):
    # 1. 序列去重 + 提前预处理所有唯一序列
    unique_sequences = list(set(original_sequences))
    total_unique = len(unique_sequences)
    print(f"原始数据共 {len(original_sequences)} 条，其中唯一序列 {total_unique} 条（仅提取一次）")

    print("⏳ 开始提前预处理所有唯一序列（计算长度+格式化）...")
    all_original_lens = [len(seq) for seq in unique_sequences]
    all_formatted = []
    for seq in unique_sequences:
        truncated = seq[:500] + seq[-500:] if len(seq) > max_len else seq
        formatted = ' '.join(list(truncated))
        formatted = re.sub(r"[UZOB]", "X", formatted)
        all_formatted.append(formatted)
    print(f"✅ 提前预处理完成，共 {total_unique} 条序列")

    # 2. 加载断点（调用修改后的load_batch_breakpoint）
    # 【修改4】接收参数改为（特征字典、长度字典、进度）
    seq_to_feat, seq_to_len, last_index = load_batch_breakpoint()
    # 补充：以实际特征数量为准，避免进度偏差
    if len(seq_to_feat) > last_index:
        last_index = len(seq_to_feat)
    print(f"唯一序列总数量：{total_unique}，已完成：{last_index}，待处理：{total_unique - last_index}")

    # 3. 批量处理唯一序列（删除所有断点保存调用）
    for start in tqdm(range(last_index, total_unique, batch_size), desc="唯一序列特征提取"):
        end = min(start + batch_size, total_unique)
        batch_seqs = unique_sequences[start:end]
        batch_original_lens = all_original_lens[start:end]
        batch_formatted = all_formatted[start:end]

        inputs = tokenizer(
            batch_formatted,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_len + 2,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        dummy_result = outputs.last_hidden_state.sum().cpu().item()
        print(f"✅ GPU计算完成！输出总和：{dummy_result}（证明计算被触发）")
        print(f"模型输出设备：{outputs.last_hidden_state.device}")

        # 存储序列-特征映射
        for i in range(len(batch_seqs)):
            seq = batch_seqs[i]
            valid_len = (inputs["attention_mask"][i] == 1).sum().item() - 2
            residue_feat = last_hidden_state[i, 1:1 + valid_len, :].cpu().numpy()
            seq_to_feat[seq] = residue_feat
            seq_to_len[seq] = batch_original_lens[i]

        # 【修改5】删除断点保存调用（不再保存任何新断点）

        # 清理显存
        del inputs, outputs, last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()

    # 【修改6】删除"处理完成删除主断点"的步骤（因未生成新断点）
    print("✅ 唯一序列处理完成")

    return seq_to_feat, seq_to_len

# ------------------------- 6. 主流程（保持原样） -------------------------
if __name__ == "__main__":
    df = pd.read_pickle(pkl_path)
    assert original_ezy_col in df.columns and sequence_col in df.columns, "缺少必要列"
    original_sequences = df[sequence_col].tolist()

    tokenizer, model = load_prot5_model()
    seq_to_feat, seq_to_len = unique_seq_extract(original_sequences, tokenizer, model)

    df["new_ezy_feat"] = df[sequence_col].map(seq_to_feat)
    df["seq_length"] = df[sequence_col].map(seq_to_len)

    df.to_pickle(output_path)
    efficiency_gain = round((1 - len(seq_to_feat)/len(df))*100, 2)
    print(f"\n✅ 最终数据集已保存至：{output_path}")
    print(f"📊 统计：原始数据 {len(df)} 条，仅提取 {len(seq_to_feat)} 条唯一序列特征，效率提升 {efficiency_gain}%")