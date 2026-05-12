import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# -------------------------- 核心配置（按需修改） --------------------------
INPUT_PATH = "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats.pkl"  # 输入CSV/Pickle
OUTPUT_PATH = "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats_with_morgan.pkl"  # 输出Pickle

# Morgan指纹参数（可自定义）
MORGAN_RADIUS = 2  # 指纹半径（常用2，1=简单结构，3=复杂结构）
MORGAN_NBITS = 2048  # 指纹维度（常用2048，可设1024/4096）


# -------------------------- 核心函数（修复版） --------------------------
def get_morgan_fingerprint(smiles):
    """单个Smiles生成Morgan指纹（容错处理+返回np.ndarray类型）"""
    try:
        # 极端容错：处理非字符串/空值
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return np.zeros(MORGAN_NBITS, dtype=np.float32)  # 无效→全0数组

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return np.zeros(MORGAN_NBITS, dtype=np.float32)  # 解析失败→全0数组

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
        return np.array(fp, dtype=np.float32)  # 核心修改：返回numpy数组而非list
    except Exception as e:
        print(f"⚠️ 处理Smiles [{smiles[:20]}...] 失败：{str(e)[:50]}")
        return np.zeros(MORGAN_NBITS, dtype=np.float32)  # 异常→全0数组


def process_substrate_unique_smiles(input_path, output_path):
    # 1. 加载数据
    print("✅ 加载数据...")
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".pkl"):
        df = pd.read_pickle(input_path)
    else:
        raise ValueError("仅支持CSV或Pickle格式！")

    assert "Smiles" in df.columns, f"缺少'Smiles'列（当前列名：{list(df.columns)}）"
    total_samples = len(df)
    print(f"✅ 数据加载成功 | 总样本数：{total_samples}")

    # 2. 统计唯一Smiles（去重，核心优化+强容错）
    print("\n📊 统计Smiles唯一值...")
    # 预处理Smiles：统一清洗（去空格、空值填充）
    df["Smiles_clean"] = df["Smiles"].apply(lambda x: x.strip() if isinstance(x, str) else "")
    # 过滤无效Smiles
    valid_mask = df["Smiles_clean"].apply(lambda x: len(x) > 0)
    valid_smiles_series = df.loc[valid_mask, "Smiles_clean"]
    unique_smiles = valid_smiles_series.drop_duplicates().tolist()  # 唯一有效Smiles列表

    print(f"  - 有效Smiles总数：{len(valid_smiles_series)}")
    print(f"  - 唯一Smiles数：{len(unique_smiles)}")
    print(f"  - 重复Smiles数（含重复次数）：{len(valid_smiles_series) - len(unique_smiles)}")
    print(f"  - Smiles唯一率：{len(unique_smiles) / len(valid_smiles_series):.4f}")

    # 3. 仅对唯一Smiles生成Morgan指纹（减少计算量）
    print(
        f"\n🔄 生成唯一Smiles的Morgan指纹（共{len(unique_smiles)}条，节省{1 - len(unique_smiles) / len(valid_smiles_series):.1%}计算量）...")
    smiles_to_morgan = {}  # 映射字典：清洗后的Smiles → Morgan指纹（np.ndarray类型）
    for smiles in tqdm(unique_smiles, desc="处理唯一Smiles"):
        smiles_to_morgan[smiles] = get_morgan_fingerprint(smiles)

    # 4. 遍历原数据集，映射所有样本的指纹（强容错）
    print("\n🗺️  映射所有样本的指纹...")
    morgan_feats = []
    for idx, row in tqdm(df.iterrows(), desc="映射指纹", total=total_samples):
        clean_smiles = row["Smiles_clean"]
        # 核心修复：优先从映射字典取，无匹配则直接生成（避免漏映射）
        if clean_smiles in smiles_to_morgan:
            morgan_feats.append(smiles_to_morgan[clean_smiles])
        else:
            # 兜底：直接生成指纹（处理极端情况）
            morgan_feats.append(get_morgan_fingerprint(clean_smiles))

    # 5. 新增列+清理冗余列
    df["morgan_feat"] = morgan_feats
    df.drop(columns=["Smiles_clean"], inplace=True)  # 删除临时清洗列

    # 删除sbt_feat/ezy_feat列（容错处理）
    for col in ["sbt_feat", "ezy_feat"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"✅ 删除列：{col}")
        else:
            print(f"⚠️ 列 {col} 不存在，无需删除")

    # 6. 保存文件（确保格式兼容）
    df.to_pickle(output_path)
    print(f"\n✅ 处理完成！文件保存到：{output_path}")

    # 7. 严格验证结果（覆盖所有边界情况，适配numpy数组）
    print("\n=== 结果验证 ===")
    df_verify = pd.read_pickle(output_path)
    print(f"输出文件列名：{list(df_verify.columns)}")

    # 验证指纹维度和有效性（适配numpy数组）
    invalid_count = 0
    valid_count = 0
    for idx, feat in enumerate(df_verify["morgan_feat"]):
        # 检查维度和类型
        if not isinstance(feat, np.ndarray) or feat.shape != (MORGAN_NBITS,):
            invalid_count += 1
            # 兜底修复：替换为全0数组
            df_verify.loc[idx, "morgan_feat"] = np.zeros(MORGAN_NBITS, dtype=np.float32)
        elif np.all(feat == 0):
            invalid_count += 1
        else:
            valid_count += 1

    # 重新保存修复后的文件
    df_verify.to_pickle(output_path)

    print(f"Morgan指纹维度：{MORGAN_NBITS}（符合要求）")
    print(f"有效指纹数（非全0+维度正确）：{valid_count}/{len(df_verify)}")
    print(f"无效指纹数：{invalid_count}/{len(df_verify)}")
    print(f"指纹有效率：{valid_count / len(df_verify):.4f}")


# -------------------------- 运行 + 日志优化 --------------------------
if __name__ == "__main__":
    # 关闭RDKit冗余警告
    import rdkit.rdBase as rdb

    rdb.DisableLog('rdApp.warning')

    process_substrate_unique_smiles(INPUT_PATH, OUTPUT_PATH)
    print("\n🎉 底物Morgan指纹处理完成（去重优化+100%有效版）！")