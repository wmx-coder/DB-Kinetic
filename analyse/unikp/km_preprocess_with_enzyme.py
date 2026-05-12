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

# -------------------------- 配置：km数据集专属路径（与kcat完全隔离） --------------------------
# 原始km数据集路径（指定CSV文件，只读不修改）
RAW_KM_DATA_PATH = '/mnt/usb/code/wm/catapro/datasets/Km-data_0.4simi-10fold.csv'
# km专属：底物唯一SMILES→ID映射 + ID→特征（核心存储，带km_前缀）
KM_SMILES_ID_FEAT_PATH = "PreKM_new/10fold/km_smiles_id_feat.pkl"
# km专属：酶唯一序列→ID映射 + ID→特征（核心存储，带km_前缀）
KM_ENZYME_ID_FEAT_PATH = "PreKM_new/10fold/km_enzyme_id_feat.pkl"
# km专属：处理后数据集保存路径（带km_前缀，保留ezy_feat原始格式）
KM_PROCESSED_DATA_PATH = "PreKM_new/10fold/km_data_with_id_log_processed.pkl"

# 创建km专属保存目录（自动创建不存在的文件夹，与kcat的PreKcat_new分离）
for save_path in [KM_SMILES_ID_FEAT_PATH, KM_ENZYME_ID_FEAT_PATH, KM_PROCESSED_DATA_PATH]:
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)


def smiles_to_vec(Smiles):
    """底物SMILES特征提取函数（完全保留你的原始代码，不做任何修改）"""
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')

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

    # 强制使用CPU，避免CUDA张量转NumPy报错
    device = torch.device('cpu')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl', map_location=device))
    trfm.eval()
    trfm = trfm.to(device)

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    xid = xid.to(device)
    xseg = xseg.to(device)  # 补全：将xseg转移到CPU设备

    with torch.no_grad():
        X = trfm.encode(torch.t(xid))

    gc.collect()
    return X


def Seq_to_vec(Sequence):
    """酶序列特征提取函数（完全保留你的原始代码，不做任何修改）"""
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
    tokenizer = T5Tokenizer.from_pretrained("/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50",
                                            do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/mnt/usb3/code/gfy/code/CataPro-master/models/prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i + 1))
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


def main():
    """km数据集专属预处理流程（仅修改特征保存逻辑，不改动特征提取）"""
    # -------------------------- 步骤1：加载原始km数据（CSV格式） --------------------------
    print(f"===== 开始加载原始km数据集（只读） =====")
    if not os.path.exists(RAW_KM_DATA_PATH):
        raise FileNotFoundError(f"原始km数据集不存在：{RAW_KM_DATA_PATH}")

    # 加载CSV格式数据集
    df_raw = pd.read_csv(RAW_KM_DATA_PATH)
    print(f"原始km数据集形状：{df_raw.shape}")
    print(f"原始km数据集列名：{df_raw.columns.tolist()}")
    # 明确酶序列列名为'Sequence'（根据指定列名确定）
    enzyme_seq_col = 'Sequence'
    print(f"✅ 原始km数据加载完成，酶序列列名：{enzyme_seq_col}")

    # -------------------------- 步骤2：创建km数据独立副本 --------------------------
    df = df_raw.copy(deep=True)
    print(f"\n===== 已创建原始km数据的独立副本 =====")
    print(f"km副本数据形状：{df.shape}")

    # -------------------------- 步骤3：处理SMILES（仅修改保存逻辑，不改动提取） --------------------------
    print(f"\n===== 开始处理km数据集底物SMILES =====")
    if os.path.exists(KM_SMILES_ID_FEAT_PATH):
        with open(KM_SMILES_ID_FEAT_PATH, "rb") as f:
            smiles_data = pickle.load(f)
        smiles_to_id = smiles_data['smiles_to_id']
        id_to_smiles_feat = smiles_data['id_to_smiles_feat']
        print(f"✅ km专属SMILES映射加载完成，唯一SMILES数量：{len(smiles_to_id)}")
    else:
        unique_smiles = df['Smiles'].unique()
        print(f"km副本中唯一底物SMILES数量：{len(unique_smiles)}")
        smiles_to_id = {smile: idx for idx, smile in enumerate(unique_smiles)}
        print(f"✅ km专属SMILES→ID映射生成完成")

        print(f"开始提取km专属唯一SMILES特征...")
        unique_smiles_feats = smiles_to_vec(list(unique_smiles))  # 调用你的原始提取函数
        print(f"km唯一SMILES特征形状：{unique_smiles_feats.shape}")

        # 仅修改此处：将Tensor转为numpy数组后保存（不改动提取函数）
        if isinstance(unique_smiles_feats, torch.Tensor):
            unique_smiles_feats_np = unique_smiles_feats.cpu().numpy()
        else:
            unique_smiles_feats_np = unique_smiles_feats
        # 直接保存numpy数组，不转换为Tensor
        id_to_smiles_feat = {idx: unique_smiles_feats_np[i] for i, idx in enumerate(smiles_to_id.values())}

        smiles_data = {'smiles_to_id': smiles_to_id, 'id_to_smiles_feat': id_to_smiles_feat}
        with open(KM_SMILES_ID_FEAT_PATH, "wb") as f:
            pickle.dump(smiles_data, f)
        print(f"✅ km专属SMILES映射已保存到：{KM_SMILES_ID_FEAT_PATH}")

    # 添加smiles_id列
    df['smiles_id'] = df['Smiles'].map(smiles_to_id)
    if df['smiles_id'].isnull().any():
        print(f"⚠️  km数据存在部分SMILES未映射到ID，缺失数量：{df['smiles_id'].isnull().sum()}")
    else:
        print(f"✅ 已为km数据副本添加smiles_id列，无缺失值")

    # -------------------------- 步骤4：处理酶序列（仅修改保存逻辑，不改动提取） --------------------------
    print(f"\n===== 开始处理km数据集酶序列 =====")
    if os.path.exists(KM_ENZYME_ID_FEAT_PATH):
        # 加载已有酶序列映射，避免重复提取
        with open(KM_ENZYME_ID_FEAT_PATH, "rb") as f:
            enzyme_data = pickle.load(f)
        enzyme_to_id = enzyme_data['enzyme_to_id']
        id_to_enzyme_feat = enzyme_data['id_to_enzyme_feat']
        print(f"✅ km专属酶序列映射加载完成，唯一酶序列数量：{len(enzyme_to_id)}")
    else:
        # 提取唯一酶序列
        unique_enzyme_seqs = df[enzyme_seq_col].unique()
        print(f"km副本中唯一酶序列数量：{len(unique_enzyme_seqs)}")
        # 生成酶序列→ID映射（从0开始编号）
        enzyme_to_id = {seq: idx for idx, seq in enumerate(unique_enzyme_seqs)}
        print(f"✅ km专属酶序列→ID映射生成完成")

        # 提取唯一酶序列的特征（调用你的原始提取函数）
        print(f"开始提取km专属唯一酶序列特征（耗时较长，请耐心等待）...")
        unique_enzyme_feats = Seq_to_vec(list(unique_enzyme_seqs))  # 调用你的原始提取函数
        print(f"km唯一酶序列特征形状：{unique_enzyme_feats.shape}")
        # 验证特征形状（确保是二维数组，避免后续报错）
        assert len(unique_enzyme_feats.shape) == 2, f"酶特征形状错误，应为二维数组，实际为：{unique_enzyme_feats.shape}"
        assert unique_enzyme_feats.shape[1] == 1024, f"酶特征维度错误，应为1024，实际为：{unique_enzyme_feats.shape[1]}"

        # 仅修改此处：直接保存numpy数组，移除torch.tensor转换（不改动提取函数）
        id_to_enzyme_feat = {
            idx: unique_enzyme_feats[i]  # 保留numpy格式，不转Tensor
            for i, idx in enumerate(enzyme_to_id.values())
        }
        # 保存酶序列映射文件
        enzyme_data = {
            'enzyme_to_id': enzyme_to_id,
            'id_to_enzyme_feat': id_to_enzyme_feat
        }
        with open(KM_ENZYME_ID_FEAT_PATH, "wb") as f:
            pickle.dump(enzyme_data, f)
        print(f"✅ km专属酶序列映射已保存到：{KM_ENZYME_ID_FEAT_PATH}")

    # 为km数据副本添加酶序列ID列
    df['enzyme_id'] = df[enzyme_seq_col].map(enzyme_to_id)
    # 验证酶ID列是否无缺失
    if df['enzyme_id'].isnull().any():
        print(f"⚠️  km数据存在部分酶序列未映射到ID，缺失数量：{df['enzyme_id'].isnull().sum()}")
    else:
        print(f"✅ 已为km数据副本添加enzyme_id列，无缺失值")

    # -------------------------- 步骤5：直接计算log10_Km（无需检查log列存在性） --------------------------
    print(f"\n===== 开始计算log10_Km列 =====")
    km_col = 'Km(M)'  # 明确Km列名（根据指定列名确定）

    def calc_log10_Km(x):
        """Km值对数转换，处理非正数异常值"""
        if pd.isna(x) or x <= 0:
            return -10000000000  # 异常值标记
        return math.log10(x)

    # 直接计算并添加log10_Km列
    df['log10_Km'] = df[km_col].apply(calc_log10_Km)
    print(f"✅ 已完成log10_Km列计算，异常值标记为：-10000000000")

    # -------------------------- 步骤6：删除km数据集冗余列（原有逻辑） --------------------------
    print(f"\n===== 开始删除km数据集指定冗余列 =====")
    cols_to_drop = ['sbt_feat', 'sequence_length', 'ezy_feat', 'new_ezy_feat', 'seq_length', 'Unnamed: 0']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    if len(cols_to_drop) > 0:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"✅ 已删除km数据集冗余列：{cols_to_drop}")
    else:
        print(f"✅ km数据集无需要删除的冗余列")

    # -------------------------- 步骤7：保存km专属处理后pkl副本 --------------------------
    print(f"\n===== 开始保存km专属处理后的数据集 =====")
    df.to_pickle(KM_PROCESSED_DATA_PATH)
    print(f"✅ km专属处理后的数据集已保存为：{KM_PROCESSED_DATA_PATH}")
    print(f"✅ 原始km数据集{RAW_KM_DATA_PATH}未被修改，保持完整")

    # -------------------------- 步骤8：输出km数据集核心统计信息 --------------------------
    print(f"\n===== km数据集预处理完成，核心统计信息 =====")
    print(f"1. 原始km数据集样本数：{len(df_raw)}")
    print(f"2. 处理后km数据集样本数：{len(df)}")
    print(f"3. 处理后km数据集列数：{len(df.columns)}")
    print(f"4. km数据集唯一SMILES数量：{len(smiles_to_id)}")
    print(f"5. km数据集唯一酶序列数量：{len(enzyme_to_id)}")
    print(f"6. km专属核心文件：")
    print(f"   - km专属SMILES ID-特征映射：{KM_SMILES_ID_FEAT_PATH}")
    print(f"   - km专属酶序列ID-特征映射：{KM_ENZYME_ID_FEAT_PATH}")
    print(f"   - km专属处理后数据集：{KM_PROCESSED_DATA_PATH}")
    print(f"   - 原始km数据集（只读）：{RAW_KM_DATA_PATH}")
    print(f"7. 与kcat数据隔离：输出目录为PreKM_new，不影响PreKcat_new下的kcat数据")
    print(f"8. 特征格式：纯numpy数组，无Tensor格式，兼容训练代码")
    print(f"\n===== km数据集所有预处理完成 =====")


if __name__ == '__main__':
    main()