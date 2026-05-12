import pandas as pd
import os  # 用于创建输出目录（确保目录存在）

# ===================== 配置参数 =====================
# 输入文件路径（km数据集，pkl格式）
INPUT_FILE = "/mnt/usb3/wmx/km/km_with_complete_feats.pkl"
# 输出目录
OUTPUT_DIR = r"/mnt/usb3/wmx/analyse"
# 筛选阈值（论文中N=30，可根据需要修改）
N = 30

# ===================== 数据处理 =====================
# 确保输出目录存在，不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 读取Km数据集（pkl格式）
df_km = pd.read_pickle(INPUT_FILE)

# 2. 定义“单个反应”：同一酶（UniProtID）+ 同一底物（Smiles）的唯一组合
df_km["reaction_id"] = df_km["UniProtID"] + "_" + df_km["Smiles"]

# 3. 统计每个反应的总突变体数（含wild野生型）
reaction_mutant_count = df_km.groupby("reaction_id")["EnzymeType"].count()

# 4. 筛选出“突变体数≥N”的有效反应
valid_reactions_km = reaction_mutant_count[reaction_mutant_count >= N].index.tolist()
df_km_filtered = df_km[df_km["reaction_id"].isin(valid_reactions_km)].copy()

# 5. 新增“是否为突变体”标签
df_km_filtered["is_mutant"] = df_km_filtered["EnzymeType"].apply(lambda x: 1 if x != "wild" else 0)

# ===================== 结果输出 =====================
# 6. 打印筛选结果统计
print(f"=== Km 数据集突变体筛选结果（N={N}）===")
print(f"原始数据集总样本数：{len(df_km)}")
print(f"筛选后总样本数：{len(df_km_filtered)}")
print(f"有效反应数（突变体数≥{N}）：{len(valid_reactions_km)}")
print(f"筛选后突变体样本数：{df_km_filtered['is_mutant'].sum()}")
print(f"筛选后野生型样本数：{len(df_km_filtered) - df_km_filtered['is_mutant'].sum()}")

# 7. 保存筛选后的数据为PKL格式（核心修改点）
output_file = f"{OUTPUT_DIR}/km-data_filtered_N{N}.pkl"
df_km_filtered.to_pickle(output_file)  # 改为to_pickle方法
print(f"\n筛选后的数据已保存为：{output_file}")

# 8. 验证所有有效反应的样本数均≥N
reaction_sample_check = df_km_filtered.groupby("reaction_id").size()
print(f"\n所有有效反应的样本数是否均≥{N}？{all(reaction_sample_check >= N)}")