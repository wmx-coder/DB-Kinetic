import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
import pickle
import math
import os

# -------------------------- 配置：仅保留核心保存路径 --------------------------
# 原始数据集路径
DATA_PATH = '/mnt/usb/code/wm/catapro/datasets/kcat-data_0.4simi-10fold.csv'
# 酶：唯一序列→ID映射 + ID→特征（核心存储）
SEQ_ID_FEAT_PATH = "PreKcat_new/10fold/seq_id_feat.pkl"
# 底物：唯一SMILES→ID映射 + ID→特征（核心存储）
SMILES_ID_FEAT_PATH = "PreKcat_new/10fold/smiles_id_feat.pkl"
# 带ID和Log列的数据集（无特征，仅核心信息）
DATA_WITH_ID_LOG_PATH = "PreKcat_new/10fold/data_with_id_log.csv"

# 创建保存目录
for save_path in [SEQ_ID_FEAT_PATH, SMILES_ID_FEAT_PATH, DATA_WITH_ID_LOG_PATH]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

def smiles_to_vec(Smiles):
    """底物SMILES特征提取函数（保持原有逻辑不变）"""
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('/mnt/vocab.pkl')

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('/mnt/trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    """酶序列特征提取函数（保持原有逻辑不变）"""
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
    tokenizer = T5Tokenizer.from_pretrained("/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50")
    gc.collect()
    print(f"CUDA可用状态：{torch.cuda.is_available()}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print(f'正在提取第 {i + 1}/{len(sequences_Example)} 条酶序列特征')
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
    # 均值池化得到最终特征
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


def main():
    """主流程：仅存储唯一序列特征 + 数据集添加ID/Log列（无冗余）"""
    # -------------------------- 步骤1：加载原始数据集 --------------------------
    print(f"开始加载原始数据集：{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"原始数据集形状：{df.shape}")

    # -------------------------- 步骤2：酶序列：仅提取唯一序列特征+保存映射 --------------------------
    if os.path.exists(SEQ_ID_FEAT_PATH):
        print(f"\n检测到酶ID-特征文件：{SEQ_ID_FEAT_PATH}，直接加载...")
        with open(SEQ_ID_FEAT_PATH, "rb") as f:
            seq_data = pickle.load(f)
        seq_to_id = seq_data['seq_to_id']
        id_to_seq_feat = seq_data['id_to_seq_feat']
    else:
        print("\n未检测到酶ID-特征文件，开始处理唯一酶序列...")
        # 提取唯一酶序列（核心：仅对唯一序列处理）
        unique_sequences = df['Sequence'].unique()
        print(f"唯一酶序列数量：{len(unique_sequences)}")
        # 构建序列→ID映射
        seq_to_id = {seq: idx for idx, seq in enumerate(unique_sequences)}
        # 仅提取唯一序列的特征
        unique_seq_feats = Seq_to_vec(list(unique_sequences))
        # 构建ID→特征映射（仅存储唯一特征）
        id_to_seq_feat = {idx: unique_seq_feats[i] for i, idx in enumerate(seq_to_id.values())}
        # 仅保存核心映射（无冗余统计）
        seq_data = {
            'seq_to_id': seq_to_id,
            'id_to_seq_feat': id_to_seq_feat
        }
        with open(SEQ_ID_FEAT_PATH, "wb") as f:
            pickle.dump(seq_data, f)
        print(f"酶唯一特征+映射已保存到：{SEQ_ID_FEAT_PATH}")
    # 为数据集添加酶ID列
    df['seq_id'] = df['Sequence'].map(seq_to_id)
    print(f"酶ID列添加完成")

    # -------------------------- 步骤3：底物SMILES：仅提取唯一SMILES特征+保存映射 --------------------------
    if os.path.exists(SMILES_ID_FEAT_PATH):
        print(f"\n检测到底物ID-特征文件：{SMILES_ID_FEAT_PATH}，直接加载...")
        with open(SMILES_ID_FEAT_PATH, "rb") as f:
            smiles_data = pickle.load(f)
        smiles_to_id = smiles_data['smiles_to_id']
        id_to_smiles_feat = smiles_data['id_to_smiles_feat']
    else:
        print("\n未检测到底物ID-特征文件，开始处理唯一SMILES...")
        # 提取唯一SMILES（核心：仅对唯一SMILES处理）
        unique_smiles = df['Smiles'].unique()
        print(f"唯一底物SMILES数量：{len(unique_smiles)}")
        # 构建SMILES→ID映射
        smiles_to_id = {smile: idx for idx, smile in enumerate(unique_smiles)}
        # 仅提取唯一SMILES的特征
        unique_smiles_feats = smiles_to_vec(list(unique_smiles))
        # 构建ID→特征映射（仅存储唯一特征）
        id_to_smiles_feat = {idx: unique_smiles_feats[i] for i, idx in enumerate(smiles_to_id.values())}
        # 仅保存核心映射（无冗余统计）
        smiles_data = {
            'smiles_to_id': smiles_to_id,
            'id_to_smiles_feat': id_to_smiles_feat
        }
        with open(SMILES_ID_FEAT_PATH, "wb") as f:
            pickle.dump(smiles_data, f)
        print(f"底物唯一特征+映射已保存到：{SMILES_ID_FEAT_PATH}")
    # 为数据集添加底物ID列
    df['smiles_id'] = df['Smiles'].map(smiles_to_id)
    print(f"底物ID列添加完成")

    # -------------------------- 步骤4：添加logkcat列（标签预处理） --------------------------
    print("\n开始添加logkcat列...")
    df['kcat_log10'] = df['kcat(s^-1)'].apply(lambda x: math.log(x, 10) if x > 0 else -10000000000)
    print(f"logkcat列添加完成")

    # -------------------------- 步骤5：保存带ID和Log的数据集（无特征） --------------------------
    df.to_csv(DATA_WITH_ID_LOG_PATH, index=False)
    print(f"\n带ID和Log的数据集已保存到：{DATA_WITH_ID_LOG_PATH}")

    # -------------------------- 步骤6：输出核心统计 --------------------------
    print("\n==================== 核心流程完成 ====================")
    print(f"1. 唯一酶序列特征数：{len(id_to_seq_feat)}")
    print(f"2. 唯一底物SMILES特征数：{len(id_to_smiles_feat)}")
    print(f"3. 带ID/Log数据集样本数：{len(df)}")
    print(f"4. 核心文件：")
    print(f"   - 酶唯一特征+映射：{SEQ_ID_FEAT_PATH}")
    print(f"   - 底物唯一特征+映射：{SMILES_ID_FEAT_PATH}")
    print(f"   - 带ID/Log数据集：{DATA_WITH_ID_LOG_PATH}")


if __name__ == '__main__':
    main()