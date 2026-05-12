import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import os
import pickle
from tqdm import tqdm
from joblib import dump, load


# ====================== 1. 复用你提供的核心特征提取函数 ======================
def Seq_to_vec(Sequence, ProtT5_model):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained(ProtT5_model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtT5_model)
    gc.collect()
    print(f"CUDA可用: {torch.cuda.is_available()}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print(f'处理序列 {i + 1}/{len(sequences_Example)}')
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


def GetMACCSKeys(smiles_list):
    """Output: np.array, size is 167."""
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        mol = Chem.MolFromSmiles(smiles_list[0],sanitize=False)
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_str = fp.ToBitString()
        fp_array = np.array([int(i) for i in fp_str])
        final_values = np.concatenate([fp_array.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            mol = Chem.MolFromSmiles(smile,sanitize=False)
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_str = fp.ToBitString()
            fp_array = np.array([int(i) for i in fp_str])
            final_values.append(fp_array.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values


def get_molT5_embed(smiles_list, Molt5_model):
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model)
    model = T5EncoderModel.from_pretrained(Molt5_model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state.cpu()
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            input_ids = tokenizer(smile, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state.cpu()
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().numpy()
            final_values.append(embed.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values


# ====================== 2. 残基级特征提取函数（复用你的逻辑） ======================
def extract_residue_feat(sequence_list, prot5_model_path, batch_size=4, max_len=1000):
    """提取酶残基级特征（new_ezy_feat），复用你提供的残基提取逻辑"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    tokenizer = T5Tokenizer.from_pretrained(prot5_model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(prot5_model_path).to(device).eval()

    # 序列去重
    unique_sequences = list(set(sequence_list))
    seq_to_feat = {}

    # 预处理所有唯一序列
    all_formatted = []
    for seq in unique_sequences:
        # 截断超长序列
        truncated = seq[:500] + seq[-500:] if len(seq) > max_len else seq
        # 格式化序列
        formatted = ' '.join(list(truncated))
        formatted = re.sub(r"[UZOB]", "X", formatted)
        all_formatted.append(formatted)

    # 批量提取特征
    for start in tqdm(range(0, len(unique_sequences), batch_size), desc="提取残基级特征"):
        end = min(start + batch_size, len(unique_sequences))
        batch_seqs = unique_sequences[start:end]
        batch_formatted = all_formatted[start:end]

        # 编码
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

        last_hidden_state = outputs.last_hidden_state.cpu().numpy()

        # 存储残基特征
        for i in range(len(batch_seqs)):
            seq = batch_seqs[i]
            valid_len = (inputs["attention_mask"][i] == 1).sum().item() - 2
            residue_feat = last_hidden_state[i, 1:1 + valid_len, :]
            seq_to_feat[seq] = residue_feat

        # 清理显存
        del inputs, outputs, last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()

    # 映射回原始序列列表
    residue_feats = [seq_to_feat[seq] for seq in sequence_list]
    return residue_feats


# ====================== 3. 数据集特征提取主函数 ======================
def process_dataset(inp_fpath, out_fpath, prot5_model, molt5_model):
    df = pd.read_csv(inp_fpath)
    print(f"处理数据集: {inp_fpath}, 原始列名: {df.columns.tolist()}")

    # 自动找你的序列（完全不改名）
    seq_list = None
    for c in ["sequence", "sequence1", "Sequence.1"]:
        if c in df.columns:
            seq_list = df[c].values.tolist()
            break

    # 你的底物就是 reactant_smiles
    smi_list = df["reactant_smiles"].values.tolist()

    # 提取特征（你原来的函数）
    print("===== 提取酶全局平均特征 (ezy_feat) =====")
    ezy_feat = Seq_to_vec(seq_list, prot5_model)

    # print("===== 提取底物双特征 (sbt_feat) =====")
    # molt5_feat = get_molT5_embed(smi_list, molt5_model)
    # maccs_feat = GetMACCSKeys(smi_list)
    # sbt_feat = np.concatenate([molt5_feat, maccs_feat], axis=1)

    print("===== 提取酶残基级特征 (new_ezy_feat) =====")
    new_ezy_feat = extract_residue_feat(seq_list, prot5_model)

    # ===================== 关键：完全按你原来的方式赋值！！！ =====================
    df["ezy_feat"] = [vec for vec in ezy_feat]
    # df["sbt_feat"] = [vec for vec in sbt_feat]
    df["new_ezy_feat"] = new_ezy_feat

    # 保存
    df.to_pickle(out_fpath)
    print(f"✅ 处理完成！原始列 + 新增3列")
    return df

# ====================== 4. 主执行逻辑 ======================
if __name__ == "__main__":
    # 模型路径（你自己改）
    PROT5_MODEL = "/mnt/usb1/wmx/prot_t5_xl_uniref50"
    MOLT5_MODEL = "/mnt/usb1/wmx/molt5"

    # 你的 7 个数据集（自动处理）
    files = [
        # "EcTL_with_sub.csv",
        # "HIS3.csv",
        # "HIS7.csv",
        # "si-dbs_ub.csv",
        # "Ssdata_ub.csv",
        # "Tmdata_with_sub.csv"
        # "Ttdata_with_sub.csv"
    ]

    input_dir = r"/mnt/usb1/wmx/catapro/analyse/dms"
    output_dir = r"/mnt/usb1/wmx/catapro/analyse/dms"
    os.makedirs(output_dir, exist_ok=True)

    for fname in files:
        inp = os.path.join(input_dir, fname)
        out = os.path.join(output_dir, fname.replace(".csv", "_feats.pkl"))

        print("\n" + "=" * 50)
        print(f"正在处理：{fname}")
        print("=" * 50)

        process_dataset(inp, out, PROT5_MODEL, MOLT5_MODEL)

    print("\n🎉 全部 7 个数据集处理完成！每个都只多 3 列特征！")