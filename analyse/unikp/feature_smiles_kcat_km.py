import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import gc
import numpy as np
import pandas as pd
import pickle
import math
import os

# -------------------------- 配置：仅保留数据集处理相关路径 --------------------------
# 原始pkl数据集路径（只读，不修改）
RAW_DATA_PATH = '/mnt/usb3/wmx/kcat_km/kcat_km_with_log_feats.pkl'
# 底物：唯一SMILES→ID映射 + ID→特征（核心存储）
SMILES_ID_FEAT_PATH = "PreKcat_km/10fold/smiles_id_feat.pkl"
# 处理后（副本删除指定列+新增ID/Log列）的数据集保存路径（pkl格式，保留ezy_feat原始格式）
PROCESSED_DATA_PATH = "PreKcat_km/10fold/data_with_id_log_processed.pkl"

# 创建保存目录（自动创建不存在的文件夹）
for save_path in [SMILES_ID_FEAT_PATH, PROCESSED_DATA_PATH]:
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)


def smiles_to_vec(Smiles):
    """底物SMILES特征提取函数（强制CPU运行，保留原有逻辑，仅移除多余.cpu()）"""
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

    # 强制使用CPU，避免CUDA张量转NumPy报错 & 节省GPU资源
    device = torch.device('cpu')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl', map_location=device))  # 模型加载到CPU
    trfm.eval()
    trfm = trfm.to(device)

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    xid = xid.to(device)  # 张量移至CPU

    # 关闭梯度计算，节省内存 & 提升速度
    with torch.no_grad():
        X = trfm.encode(torch.t(xid))

    # 修复：X已经是numpy.ndarray，无需.cpu()转换，直接删除该调用
    gc.collect()  # 释放内存
    return X


def main():
    """主流程：仅负责数据集处理（单一职责），完全保留ezy_feat原始格式
    1. 加载原始pkl数据（只读）
    2. 创建独立副本（不影响原始数据）
    3. 处理SMILES：生成唯一ID+特征映射
    4. 为副本添加SMILES ID列
    5. 处理log10_kcat_over_Km列（复用/新增）
    6. 删除指定冗余列
    7. 保存处理后的pkl副本（保留ezy_feat原始格式）
    """
    # -------------------------- 步骤1：加载原始数据（只读，不做任何修改） --------------------------
    print(f"===== 开始加载原始数据集（只读） =====")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"原始数据集不存在：{RAW_DATA_PATH}")

    df_raw = pd.read_pickle(RAW_DATA_PATH)
    print(f"原始数据集形状：{df_raw.shape}")
    print(f"原始数据集列名：{df_raw.columns.tolist()}")
    print(f"ezy_feat列数据类型：{type(df_raw['ezy_feat'].iloc[0])}")
    print(
        f"ezy_feat列形状示例：{df_raw['ezy_feat'].iloc[0].shape if isinstance(df_raw['ezy_feat'].iloc[0], np.ndarray) else len(df_raw['ezy_feat'].iloc[0])}")
    print(f"✅ 原始数据加载完成，ezy_feat保持原始格式")

    # -------------------------- 步骤2：创建数据独立副本（所有操作均在副本上进行） --------------------------
    df = df_raw.copy(deep=True)  # deep=True确保完全独立，与原始数据解耦，保留ezy_feat原始格式
    print(f"\n===== 已创建原始数据的独立副本 =====")
    print(f"副本数据形状：{df.shape}")
    print(f"副本ezy_feat列数据类型：{type(df['ezy_feat'].iloc[0])}")
    print(f"✅ 后续所有处理仅针对副本，ezy_feat格式不变")

    # -------------------------- 步骤3：底物SMILES处理：生成唯一ID+特征映射 --------------------------
    print(f"\n===== 开始处理底物SMILES =====")
    if os.path.exists(SMILES_ID_FEAT_PATH):
        # 已存在映射文件，直接加载（避免重复提取特征，节省时间）
        print(f"检测到底物ID-特征映射文件：{SMILES_ID_FEAT_PATH}，直接加载...")
        with open(SMILES_ID_FEAT_PATH, "rb") as f:
            smiles_data = pickle.load(f)
        smiles_to_id = smiles_data['smiles_to_id']
        id_to_smiles_feat = smiles_data['id_to_smiles_feat']
        print(f"✅ 加载完成，唯一SMILES数量：{len(smiles_to_id)}")
    else:
        # 不存在映射文件，重新生成
        print(f"未检测到底物ID-特征映射文件，开始生成...")
        # 提取副本中的唯一SMILES
        unique_smiles = df['Smiles'].unique()
        print(f"副本中唯一底物SMILES数量：{len(unique_smiles)}")

        # 生成SMILES→ID映射（从0开始编号）
        smiles_to_id = {smile: idx for idx, smile in enumerate(unique_smiles)}
        print(f"✅ SMILES→ID映射生成完成")

        # 提取唯一SMILES的特征（仅提取一次，避免重复计算）
        print(f"开始提取唯一SMILES特征（CPU运行，耗时较长，请耐心等待）...")
        unique_smiles_feats = smiles_to_vec(list(unique_smiles))
        print(f"唯一SMILES特征形状：{unique_smiles_feats.shape}")
        print(f"SMILES特征数据类型：{unique_smiles_feats.dtype}")

        # 生成ID→特征映射（与ezy_feat格式一致，均为numpy数组）
        id_to_smiles_feat = {idx: unique_smiles_feats[i] for i, idx in enumerate(smiles_to_id.values())}
        print(f"✅ ID→特征映射生成完成，特征格式与ezy_feat一致")

        # 保存映射文件（供后续训练使用）
        smiles_data = {
            'smiles_to_id': smiles_to_id,
            'id_to_smiles_feat': id_to_smiles_feat
        }
        with open(SMILES_ID_FEAT_PATH, "wb") as f:
            pickle.dump(smiles_data, f)
        print(f"✅ 底物ID-特征映射已保存到：{SMILES_ID_FEAT_PATH}")

    # -------------------------- 步骤4：为数据副本添加SMILES ID列 --------------------------
    df['smiles_id'] = df['Smiles'].map(smiles_to_id)
    # 验证SMILES ID是否添加成功（无缺失值）
    if df['smiles_id'].isnull().any():
        print(f"⚠️  存在部分SMILES未映射到ID，缺失数量：{df['smiles_id'].isnull().sum()}")
    else:
        print(f"✅ 已为数据副本添加smiles_id列，无缺失值")

    # -------------------------- 步骤5：处理log10_kcat_over_Km列（复用/新增） --------------------------
    print(f"\n===== 开始处理log10_kcat_over_Km列 =====")
    if 'log10_kcat_over_Km' in df.columns:
        # 复用已有列，确保列名统一，不修改原始值
        df['log10_kcat_over_Km'] = df['log10_kcat_over_Km'].copy()
        print(f"✅ 已复用现有log10_kcat_over_Km列，值保持不变")
    else:
        # 无现有列，基于kcat/Km计算
        print(f"未检测到log10_kcat_over_Km列，基于kcat/Km计算...")

        # ========== 核心修改部分 ==========
        def calc_log10_kcat_over_Km(row):
            """计算log10(kcat/Km)，增加Km值校验"""
            # 提取kcat和Km值（请确认列名是否为kcat(s^-1)和Km(M)，如果不是请修改）
            kcat = row['kcat(s^-1)']
            km = row['Km(M)']  # 假设Km列名为Km(M)，如果你的列名不同请替换

            # 异常值判断：kcat/Km <=0 或 Km=0（避免除零错误）均标记为异常
            if kcat <= 0 or km <= 0 or (kcat / km) <= 0:
                return -10000000000  # 保持异常值标记与后续训练逻辑一致
            return math.log10(kcat / km)

        # 应用函数（使用apply(axis=1)处理行数据，同时获取kcat和Km）
        df['log10_kcat_over_Km'] = df.apply(calc_log10_kcat_over_Km, axis=1)
        print(f"✅ 已新增log10_kcat_over_Km列，异常值标记为：-10000000000")
        # ========== 核心修改结束 ==========

    # -------------------------- 步骤6：删除指定冗余列 --------------------------
    print(f"\n===== 开始删除指定冗余列 =====")
    # 要删除的列列表（可根据需求调整）
    cols_to_drop = ['sbt_feat', 'sequence_length', 'new_ezy_feat', 'seq_length']
    # 过滤掉不存在的列，避免删除报错
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    if len(cols_to_drop) > 0:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"✅ 已删除冗余列：{cols_to_drop}")
    else:
        print(f"✅ 无需要删除的冗余列")

    # 打印处理后副本的信息，验证ezy_feat格式不变
    print(f"处理后数据副本形状：{df.shape}")
    print(f"处理后数据副本列名：{df.columns.tolist()}")
    print(f"处理后ezy_feat列数据类型：{type(df['ezy_feat'].iloc[0])}")
    print(
        f"处理后ezy_feat列形状示例：{df['ezy_feat'].iloc[0].shape if isinstance(df['ezy_feat'].iloc[0], np.ndarray) else len(df['ezy_feat'].iloc[0])}")
    print(f"✅ ezy_feat保持原始格式，未做任何修改")

    # -------------------------- 步骤7：保存处理后的pkl副本（保留ezy_feat原始格式） --------------------------
    print(f"\n===== 开始保存处理后的数据集 =====")
    # pkl格式保存，完整保留ezy_feat的numpy数组格式，无丢失/变形
    df.to_pickle(PROCESSED_DATA_PATH)
    print(f"✅ 处理后的数据集已保存为：{PROCESSED_DATA_PATH}")
    print(f"✅ 保存格式为pkl，完整保留ezy_feat原始格式")
    print(f"✅ 原始数据集{RAW_DATA_PATH}未被修改，保持完整")

    # -------------------------- 步骤8：输出核心统计信息 --------------------------
    print(f"\n===== 数据集处理完成，核心统计信息 =====")
    print(f"1. 原始数据集样本数：{len(df_raw)}")
    print(f"2. 处理后数据集样本数：{len(df)}")
    print(f"3. 处理后数据集列数：{len(df.columns)}")
    print(f"4. 唯一SMILES数量：{len(smiles_to_id)}")
    print(f"5. ezy_feat格式：{type(df['ezy_feat'].iloc[0])}（原始格式保留）")
    print(f"6. 核心文件：")
    print(f"   - 底物ID-特征映射：{SMILES_ID_FEAT_PATH}")
    print(f"   - 处理后数据集（保留ezy_feat原始格式）：{PROCESSED_DATA_PATH}")
    print(f"   - 原始数据集（只读）：{RAW_DATA_PATH}")
    print(f"\n===== 所有处理完成 =====")


if __name__ == '__main__':
    main()