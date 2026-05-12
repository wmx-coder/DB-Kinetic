import pandas as pd
import numpy as np

# -------------------------- 配置路径 --------------------------
# 你的已有Pickle文件路径（含sbt_feat列）
INPUT_PKL_PATH = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_v1_with_fold.pkl"
# 拆分后的输出文件路径（可覆盖原文件，或保存为新文件）
OUTPUT_PKL_PATH = "/mnt/usb3/code/wm/data/kcat_km/kcat_km_v1_with_molt_maccs.pkl"


# -------------------------- 核心拆分逻辑 --------------------------
def split_sbt_feat(input_path, output_path):
    # 1. 加载已有数据集
    print("✅ 加载已有数据集...")
    df = pd.read_pickle(input_path)

    # 检查sbt_feat列是否存在，且维度正确
    assert "sbt_feat" in df.columns, "数据集中没有找到sbt_feat列！"
    sample_feat = df["sbt_feat"].iloc[0]
    assert len(sample_feat) == 935, f"sbt_feat维度错误（应为935维，实际为{len(sample_feat)}维）"
    print(f"✅ 验证成功：sbt_feat列存在，维度为935维")

    # 2. 拆分出molt5_feat（前768维）和maccs_feat（后167维）
    print("🔄 开始拆分特征...")
    df["molt5_feat"] = df["sbt_feat"].apply(lambda x: x[:768])  # 前768维：MolT5特征
    df["maccs_feat"] = df["sbt_feat"].apply(lambda x: x[-167:])  # 后167维：MACCS特征

    # 3. 删除原始的sbt_feat列（核心修改）
    df.drop(columns=["sbt_feat"], inplace=True)
    df.drop(columns=["ezy_feat"], inplace=True)
    print("✅ 已删除原始sbt_feat列，避免数据冗余")

    # 4. 验证拆分结果
    print("✅ 拆分完成，验证维度：")
    print(f"  - molt5_feat维度：{len(df['molt5_feat'].iloc[0])}（应为768维）")
    print(f"  - maccs_feat维度：{len(df['maccs_feat'].iloc[0])}（应为167维）")
    print(f"  - sbt_feat列状态：{'存在' if 'sbt_feat' in df.columns else '已删除'}")

    # 5. 保存拆分后的数据集
    df.to_pickle(output_path)
    print(f"\n✅ 拆分后文件已保存到：{output_path}")
    print(f"✅ 最终数据集包含列：{list(df.columns)}")


# -------------------------- 运行拆分 + 验证 --------------------------
if __name__ == "__main__":
    split_sbt_feat(INPUT_PKL_PATH, OUTPUT_PKL_PATH)

    # 验证拆分后的文件
    print("\n" + "=" * 60)
    print("验证拆分结果（前2条样本）")
    print("=" * 60)
    df_final = pd.read_pickle(OUTPUT_PKL_PATH)
    for i in range(min(2, len(df_final))):
        print(f"\n第{i + 1}条样本：")
        print(f"  - molt5_feat：{df_final['molt5_feat'].iloc[i].shape}")
        print(f"  - maccs_feat：{df_final['maccs_feat'].iloc[i].shape}")
        print(f"  - sbt_feat：{'已删除' if 'sbt_feat' not in df_final.columns else df_final['sbt_feat'].iloc[i].shape}")

print("\n🎉 特征拆分完成！已新增molt5_feat和maccs_feat列，并删除原始sbt_feat列～")