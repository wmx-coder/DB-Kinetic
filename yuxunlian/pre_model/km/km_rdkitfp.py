import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
input_pkl_path = "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats.pkl"
output_pkl_path = "/mnt/usb3/code/wm/data/km_data/km_with_complete_feats_with_rdkitfp.pkl"
fp_dim = 2048  # RDKit指纹维度
min_path = 2  # 最小路径长度（默认2）
max_path = 7  # 最大路径长度（默认7）


# -------------------------- 核心函数：生成有效RDKit Fingerprint --------------------------
def generate_rdkit_fp(smiles):
    """
    输入SMILES，生成2048维RDKit拓扑指纹（路径长度2-7）
    返回：2048维np.ndarray（无效时返回全0数组）
    """
    try:
        # 1. 校验SMILES有效性（过滤非字符串/空值）
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return np.zeros(fp_dim, dtype=np.float32)

        # 2. RDKit解析SMILES（容错解析失败场景）
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return np.zeros(fp_dim, dtype=np.float32)

        # 3. 生成RDKit原生拓扑指纹（核心操作）
        fp = Chem.RDKFingerprint(
            mol,
            fpSize=fp_dim,
            minPath=min_path,
            maxPath=max_path
        )
        fp_arr = np.array(fp, dtype=np.float32)

        # 4. 校验指纹维度（避免配置错误）
        if fp_arr.shape != (fp_dim,):
            print(f"⚠️ 指纹维度错误：预期{fp_dim}维，实际{fp_arr.shape[0]}维 | SMILES={smiles[:50]}")
            return np.zeros(fp_dim, dtype=np.float32)

        # 5. 日志提示全0指纹
        if np.all(fp_arr == 0):
            print(f"⚠️  合法SMILES生成全0指纹：{smiles[:50]}")

        return fp_arr
    except Exception as e:
        print(f"❌ 处理SMILES失败：{smiles[:50]} | 错误：{str(e)[:100]}")
        return np.zeros(fp_dim, dtype=np.float32)


# -------------------------- 主函数：生成指纹并保存 --------------------------
def main():
    # 1. 加载原文件
    print(f"✅ 正在加载原文件：{input_pkl_path}")
    df = pd.read_pickle(input_pkl_path)
    print(f"✅ 文件加载完成 | 总行数：{len(df)}")

    # 2. 检查Smiles列是否存在
    if "Smiles" not in df.columns:
        print(f"❌ 错误：文件中无 'Smiles' 列！当前列名：{list(df.columns)}")
        return

    # 3. 提取唯一SMILES（去重优化）
    print("\n=== 提取唯一SMILES（优化计算效率）===")
    df["Smiles_clean"] = df["Smiles"].apply(lambda x: x.strip() if isinstance(x, str) else "")
    valid_smiles = df[df["Smiles_clean"].str.len() > 0]["Smiles_clean"].unique()
    print(f"总样本数：{len(df)} | 唯一有效SMILES数：{len(valid_smiles)}")

    # 4. 仅为唯一SMILES生成指纹
    smiles_to_fp = {}
    for smiles in tqdm(valid_smiles, desc="生成唯一SMILES的RDKit指纹"):
        smiles_to_fp[smiles] = generate_rdkit_fp(smiles)

    # 5. 映射指纹到原数据集
    print("\n=== 映射指纹到原数据集 ===")

    def map_fp(row):
        clean_smiles = row["Smiles_clean"]
        return smiles_to_fp.get(clean_smiles, np.zeros(fp_dim, dtype=np.float32))

    df["rdkitfp_feat"] = df.apply(map_fp, axis=1)
    df.drop(columns=["Smiles_clean"], inplace=True)

    # 6. 统计有效指纹
    def is_valid_fp(fp):
        if not isinstance(fp, np.ndarray) or len(fp) != fp_dim:
            return False
        return not np.all(fp == 0)

    df["is_valid_fp"] = df["rdkitfp_feat"].apply(is_valid_fp)
    valid_count = df["is_valid_fp"].sum()
    invalid_count = len(df) - valid_count

    # 7. 打印统计结果
    print("\n=== 指纹生成统计 ===")
    print(f"有效指纹行数：{valid_count}/{len(df)}")
    print(f"无效指纹行数：{invalid_count}/{len(df)}")
    print(f"指纹有效率：{valid_count / len(df):.4f}")

    # 8. 删除冗余列
    for col in ["sbt_feat", "ezy_feat"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"✅ 删除冗余列：{col}")
        else:
            print(f"⚠️ 列 {col} 不存在，无需删除")

    # 9. 保存新文件
    df.to_pickle(output_pkl_path, compression=None)
    print(f"\n✅ 新文件已保存：{output_pkl_path}")

    # 10. 验证前5条指纹
    print("\n=== 前5条指纹验证 ===")
    for idx in range(min(5, len(df))):
        fp = df.loc[idx, "rdkitfp_feat"]
        smiles = df.loc[idx, "Smiles"]
        non_zero_count = np.count_nonzero(fp)
        print(f"行{idx}：SMILES={smiles[:30]} | 非0特征数={non_zero_count}/{fp_dim} | 类型={type(fp)}")


if __name__ == "__main__":
    import rdkit.rdBase as rdb

    rdb.DisableLog('rdApp.warning')
    main()