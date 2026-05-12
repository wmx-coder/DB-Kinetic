# # import pandas as pd
# #
# # # 读取数据
# # file_path = r"D:\fwq\catapro\analyse\dms\ALL_DATASET_SCC_FINAL.xlsx"
# # df = pd.read_excel(file_path)
# #
# # # 只保留 final 类型
# # df_final = df[df["Type"] == "final"].copy()
# #
# # # ===================== 1. 查看有哪些模型 =====================
# # models = df_final["Model"].unique()
# # print("✅ 检测到的模型：", list(models))
# # print("=" * 80)
# #
# # # ===================== 2. 每个模型有多少条数据 =====================
# # print("📊 每个模型的数据条数：")
# # for m in models:
# #     cnt = len(df_final[df_final["Model"] == m])
# #     print(f"   {m}：{cnt} 条")
# #
# # print("=" * 80)
# #
# # # ===================== 3. 按【数据集 + 模型】提取结果 =====================
# # datasets = ["EcTL", "HIS3", "HIS7", "si-dbs_ub", "Ssdata_ub", "Ttdata"]
# #
# # result_list = []
# #
# # for dataset in datasets:
# #     for model in models:
# #         sub = df_final[
# #             (df_final["Dataset"] == dataset) &
# #             (df_final["Model"] == model)
# #             ]
# #
# #         kcat = sub[sub["Task"] == "kcat"]["SCC"].values
# #         km = sub[sub["Task"] == "Km"]["SCC"].values
# #         kcatkm = sub[sub["Task"] == "kcat/Km"]["SCC"].values
# #
# #         kcat_val = kcat[0] if len(kcat) > 0 else None
# #         km_val = km[0] if len(km) > 0 else None
# #         kcatkm_val = kcatkm[0] if len(kcatkm) > 0 else None
# #
# #         result_list.append({
# #             "Dataset": dataset,
# #             "Model": model,
# #             "kcat_final": round(kcat_val, 4) if kcat_val else None,
# #             "Km_final": round(km_val, 4) if km_val else None,
# #             "kcat/Km_final": round(kcatkm_val, 4) if kcatkm_val else None
# #         })
# #
# # # 转成表格
# # result_df = pd.DataFrame(result_list)
# #
# # # ===================== 4. 打印漂亮结果 =====================
# # print("🔥 【每个数据集 × 每个模型 的 final SCC 结果】\n")
# # for dataset in datasets:
# #     print(f"🔹 数据集：{dataset}")
# #     sub_df = result_df[result_df["Dataset"] == dataset]
# #     for _, row in sub_df.iterrows():
# #         print(f"   模型 {row['Model']}：")
# #         print(f"      kcat  = {row['kcat_final']}")
# #         print(f"      Km    = {row['Km_final']}")
# #         print(f"      kcat/Km = {row['kcat/Km_final']}")
# #     print("-" * 60)
# #
# # # # ===================== 5. 导出 Excel（最干净版本） =====================
# # # result_df.to_excel(r"D:\fwq\catapro\analyse\dms\FINAL_SCC_DATASET_MODEL.xlsx", index=False)
# # # print("\n✅ 已导出：FINAL_SCC_DATASET_MODEL.xlsx（可直接用于论文表格）")
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # ===================== 基础设置 =====================
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['svg.fonttype'] = 'path'
#
# # 读取数据 (假设数据结构不变)
# file_path = r"D:\fwq\catapro\analyse\dms\ALL_DATASET_SCC_FINAL.xlsx"
# df = pd.read_excel(file_path)
# df_final = df[df["Type"] == "final"].copy()
#
# # 固定顺序
# datasets = ["EcTL", "HIS3", "HIS7", "si-dbs_ub", "Ssdata_ub", "Ttdata"]
# letters = ["a", "b", "c", "d", "e", "f"]
# tasks = ["kcat", "kcat/Km"]
# models = ["CataPro", "DB-Kinetic", "Eitlem"]
#
# # 配色
# colors = ["#ff9999", "#66b3ff", "#99ff99"]
#
# # ===================== 绘图：2行3列 =====================
# # 增加 figsize 的高度以容纳底部的图例
# fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
# axes = axes.flatten()
#
# x = np.arange(len(tasks))
# width = 0.25
#
# for idx, (ds, label) in enumerate(zip(datasets, letters)):
#     ax = axes[idx]
#     ds_data = df_final[df_final["Dataset"] == ds]
#
#     # 画三个模型的分组柱子
#     bars = []
#     for i, model in enumerate(models):
#         m_data = ds_data[ds_data["Model"] == model]
#         # 预防数据缺失导致报错
#         try:
#             values = [m_data[m_data["Task"] == t]["SCC"].values[0] for t in tasks]
#         except IndexError:
#             values = [0, 0]
#
#         b = ax.bar(x + (i - 1) * width, values, width, label=model, color=colors[i])
#         bars.append(b)
#
#     # ===================== 修改点 1：标记到框外，不加括号 =====================
#     # 使用 set_title 的 loc='left' 可以很方便地放在框外左上角，或者用 text 配合 transform
#     ax.set_title(label, loc='left', fontsize=20, fontweight='bold', pad=10)
#
#     # 图表样式
#     ax.set_xticks(x)
#     ax.set_xticklabels(tasks, fontsize=12)
#     ax.set_ylabel('SCC', fontsize=13)
#     ax.set_ylim(-0.35, 0.55)
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
#
# # ===================== 修改点 2：图例放到最下面 =====================
# # 从最后一个子图获取图例句柄
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14, frameon=False)
#
# # 调整整体布局，为顶部标题和底部图例留出空间
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3)
#
# # ===================== 保存为 SVG =====================
# output_path = r"D:\fwq\catapro\analyse\dms\6_dataset_abcdef.svg"
# plt.savefig(output_path, format="svg", bbox_inches='tight')
# plt.show()
#
# print(f"✅ 已生成：字母已移至框外并去括号，图例已置于底部。保存路径：{output_path}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== 基础设置 =====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'path'

# 读取数据
file_path = r"D:\fwq\catapro\analyse\dms\ALL_DATASET_SCC_FINAL.xlsx"
df = pd.read_excel(file_path)
df_final = df[df["Type"] == "final"].copy()

# 4个数据集
datasets = ["EcTL", "HIS3", "si-dbs_ub", "Ttdata"]
letters = ["a", "b", "c", "d"]
tasks = ["kcat", "kcat/Km"]

# ===================== 核心：读取用真名，图例显示别名 =====================
# 【读取时用的真实名称】
model_real = ["DB-Kinetic", "CataPro", "Eitlem"]
# 【你要在图例上显示的名称】
model_show = ["DB-Kinetic", "CataPro", "EITLEM"]

# 经典matplotlib期刊标配三色，耐看永不丑
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# colors = ["#3D8EB9", "#FF8A65", "#66BB6A"]
colors = ["#87B8F2", "#FFB98F", "#A2E0A2"]

# ===================== 绘图 =====================
fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
axes = axes.flatten()

x = np.arange(len(tasks))
width = 0.25

for idx, (ds, label) in enumerate(zip(datasets, letters)):
    ax = axes[idx]
    ds_data = df_final[df_final["Dataset"] == ds]

    for i, (real_name, show_name) in enumerate(zip(model_real, model_show)):
        m_data = ds_data[ds_data["Model"] == real_name]
        try:
            values = [m_data[m_data["Task"] == t]["SCC"].values[0] for t in tasks]
        except IndexError:
            values = [0, 0]

        ax.bar(x + (i - 1) * width, values, width, label=show_name, color=colors[i])

    # 子图 abc 标记
    ax.set_title(label, loc='left', fontsize=20, fontweight='bold', pad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylabel('SCC', fontsize=13)
    ax.set_ylim(-0.35, 0.55)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# 底部图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3)

output_path = r"D:\fwq\catapro\analyse\dms\4_dataset_FINAL.svg"
plt.savefig(output_path, format="svg", bbox_inches='tight')
plt.show()

print("✅ 换成经典期刊蓝橙绿，耐看高级不花哨")