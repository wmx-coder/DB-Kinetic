import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import re
import gc
import math
import numpy as np
import pandas as pd
import os
import pickle
from transformers import T5EncoderModel, T5Tokenizer

# -------------------------- 配置参数 --------------------------
DATASET_PATH = r"/mnt/usb1/wmx/catapro/datasets/kcat-over-Km-data_0.4simi-10fold.csv"
FEATURE_ROOT = "/mnt/usb3/wmx/kcat_km/saved_features"
SMILES_FEAT_DIR = os.path.join(FEATURE_ROOT, "smiles_features")
ENZYME_FEAT_DIR = os.path.join(FEATURE_ROOT, "enzyme_seq_features")
ID_DATASET_PATH = "./id_based_dataset.pkl"


# -------------------------- 工具函数 --------------------------
def create_dir(dir_path):
    """创建文件夹，若已存在则不报错"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


# -------------------------- SMILES特征提取（彻底适配numpy输出）--------------------------
def smiles_to_vec(Smiles):
    """
    批量SMILES特征提取（适配encode返回numpy数组的情况）
    """
    # 固定参数（和原始逻辑一致）
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4

    # 加载词汇表和模型
    vocab = WordVocab.load_vocab('vocab.pkl')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    # 修复torch.load安全警告，加载模型权重
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl', map_location='cpu', weights_only=True))
    trfm.eval()  # 推理模式

    def get_inputs(sm):
        """处理单个SMILES为模型输入"""
        seq_len = 220
        sm = sm.split()
        # 超长截断（原始逻辑）
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        # 转换为id
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        # 补全到固定长度
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        return ids

    # 处理所有SMILES为输入id
    x_id = [get_inputs(split(sm)) for sm in Smiles]
    x_id = torch.tensor(x_id)  # 转换为张量供模型输入

    # 核心：encode输出已为numpy数组，直接返回
    with torch.no_grad():  # 禁用梯度，节省内存
        X = trfm.encode(torch.t(x_id))  # 返回numpy数组

    return X


# -------------------------- 单个SMILES处理 --------------------------
def smiles_to_vec_single(sm):
    """单个SMILES特征提取（纯numpy操作，无张量方法）"""
    # 调用批量函数，传入长度为1的列表
    X = smiles_to_vec([sm])
    # 直接挤压维度，返回一维特征向量
    return X.squeeze()


# -------------------------- 酶序列特征提取（保留原始逻辑，优化内存）--------------------------
def Seq_to_vec(Sequence):
    """批量酶序列特征提取（原始逻辑+内存优化）"""
    # 超长序列截断
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]

    # 序列添加空格分隔
    sequences_Example = []
    for seq in Sequence:
        zj = ' '.join(list(seq))  # 简化原始的循环拼接逻辑，结果一致
        sequences_Example.append(zj)

    # 加载模型和tokenizer
    tokenizer = T5Tokenizer.from_pretrained("/mnt/usb1/wmx/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/mnt/usb1/wmx/prot_t5_xl_uniref50")
    gc.collect()

    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 提取特征
    features = []
    for i, seq in enumerate(sequences_Example):
        print(f'处理酶序列 {i + 1}/{len(sequences_Example)}')
        # 替换特殊字符
        seq_clean = [re.sub(r"[UZOB]", "X", seq)]

        with torch.no_grad():  # 禁用梯度，关键！
            # 编码序列
            ids = tokenizer.batch_encode_plus(seq_clean, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            # 模型推理
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        # 转换为numpy并截断padding部分
        embedding_np = embedding.last_hidden_state.cpu().numpy()
        seq_len = (attention_mask[0] == 1).sum()
        seq_emd = embedding_np[0][:seq_len - 1]
        features.append(seq_emd)

    # 平均归一化（原始三重循环逻辑）
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            avg_val = np.mean([features[i][j][k] for j in range(len(features[i]))])
            features_normalize[i][k] = avg_val

    return features_normalize


# -------------------------- 单个酶序列处理 --------------------------
def enzyme_seq_to_vec_single(seq):
    """单个酶序列特征提取"""
    features_normalize = Seq_to_vec([seq])
    return features_normalize.squeeze()


# -------------------------- 核心：处理唯一项并保存特征 --------------------------
def process_unique_items(items, item_type, save_dir):
    """
    处理唯一项，提取特征并保存为npy文件
    """
    # 去重
    unique_items = list(pd.unique(items))
    print(f"发现 {len(unique_items)} 个唯一的{item_type}")
    create_dir(save_dir)

    # 遍历处理
    for item_id, item in enumerate(unique_items):
        try:
            # 提取特征
            if item_type == "smiles":
                feat = smiles_to_vec_single(item)
            elif item_type == "enzyme":
                feat = enzyme_seq_to_vec_single(item)
            else:
                raise ValueError("item_type仅支持 'smiles' 或 'enzyme'")

            # 保存特征
            np.save(os.path.join(save_dir, f"{item_id}.npy"), feat)

            # 进度提示
            if (item_id + 1) % 10 == 0:
                print(f"已处理 {item_id + 1}/{len(unique_items)} 个{item_type}")

        except Exception as e:
            print(f"⚠️  处理{item_type} ID {item_id} 失败：{str(e)[:100]}，跳过")
            continue

    # 构建原始item对应的ID映射
    item_to_id = {item: idx for idx, item in enumerate(unique_items)}
    original_item_ids = [item_to_id[item] for item in items]

    return original_item_ids


# -------------------------- 主函数 --------------------------
def main():
    # 创建文件夹
    create_dir(FEATURE_ROOT)
    create_dir(SMILES_FEAT_DIR)
    create_dir(ENZYME_FEAT_DIR)

    # 读取数据集
    print("📥 读取数据集...")
    df = pd.read_csv(DATASET_PATH, index_col=0)
    # 验证列
    required_cols = ['Sequence', 'Smiles', 'kcat(s^-1)', 'Km(M)', 'fold']
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"数据集缺少列：{col}")

    # 提取核心数据
    enzyme_list = df['Sequence'].values
    smiles_list = df['Smiles'].values
    kcat = df['kcat(s^-1)'].values
    km = df['Km(M)'].values
    fold_list = df['fold'].values
    print(f"✅ 数据集读取完成，共 {len(enzyme_list)} 个样本")

    # 计算kcat/km并对数转换
    print("📊 处理标签（kcat/km + 对数转换）...")
    label_processed = []
    for kcat_val, km_val in zip(kcat, km):
        if kcat_val <= 0 or km_val <= 0:
            print(f"⚠️  跳过无效标签：kcat={kcat_val}, Km={km_val}")
            continue
        label_processed.append(math.log10(kcat_val / km_val))
    label_processed = np.array(label_processed)
    print(f"✅ 标签处理完成，有效标签数：{len(label_processed)}")

    # 处理SMILES特征
    print("🧪 处理SMILES特征...")
    smiles_ids = process_unique_items(smiles_list, "smiles", SMILES_FEAT_DIR)

    # 处理酶序列特征
    print("🧬 处理酶序列特征...")
    enzyme_ids = process_unique_items(enzyme_list, "enzyme", ENZYME_FEAT_DIR)

    # 构建ID数据集（对齐长度）
    print("📦 构建ID化数据集...")
    valid_len = min(len(smiles_ids), len(enzyme_ids), len(label_processed), len(fold_list))
    id_dataset = pd.DataFrame({
        "smiles_id": smiles_ids[:valid_len],
        "enzyme_id": enzyme_ids[:valid_len],
        "label_processed": label_processed[:valid_len],
        "fold": fold_list[:valid_len]
    })

    # 保存数据集
    with open(ID_DATASET_PATH, "wb") as f:
        pickle.dump(id_dataset, f)

    # 输出汇总
    print("\n🎉 数据处理完成！汇总信息：")
    print(f"  - SMILES特征文件数：{len(os.listdir(SMILES_FEAT_DIR))}")
    print(f"  - 酶序列特征文件数：{len(os.listdir(ENZYME_FEAT_DIR))}")
    print(f"  - ID数据集路径：{ID_DATASET_PATH}（样本数：{len(id_dataset)}）")
    if len(label_processed) > 0:
        print(f"  - 标签范围：{np.min(label_processed):.4f} ~ {np.max(label_processed):.4f}")


if __name__ == '__main__':
    main()