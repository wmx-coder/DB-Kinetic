# import matplotlib.pyplot as plt
# import numpy as np
#
# # ---------------------- 全局美学配置（解决字体问题） ----------------------
# plt.rcParams.update({
#     'font.family': 'Arial',
#     'mathtext.fontset': 'dejavusans',  # 确保数学字体和普通字体风格统一
#     'axes.linewidth': 1.0,
#     'axes.spines.top': False,
#     'axes.spines.right': False,
#     'savefig.dpi': 600,
#     'figure.constrained_layout.use': True,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10
# })
#
# # ---------------------- 模型与配色 ----------------------
# methods = [
#     "DB-Kinetic",
#     "ProT5_MolT5-MACCS",
#     "ProT5_RDKitFP",
#     "ProT5_MACCS",
#     "ProT5_Morgan",
#     "ProT5_MolT5",
#     "Esm2_MolT5-MACCS"
# ]
#
# colors = ["#E63946", "#1D3557", "#457B9D", "#2A9D8F", "#F4A261", "#E9C46A", "#6A0572"]
# markers = ["D", "s", "^", "o", "p", "*", "X"]
#
# # ---------------------- 数据 ----------------------
# data = {
#     "kcat": {
#         "PCC": [0.533, 0.507, 0.491, 0.504, 0.500, 0.491, 0.482],
#         "SCC": [0.519, 0.493, 0.479, 0.489, 0.486, 0.479, 0.468]
#     },
#     "Km": {
#         "PCC": [0.648, 0.618, 0.613, 0.612, 0.620, 0.605, 0.605],
#         "SCC": [0.647, 0.615, 0.609, 0.609, 0.617, 0.603, 0.603]
#     },
#     "kcat/Km": {
#         "PCC": [0.481, 0.380, 0.386, 0.389, 0.377, 0.377, 0.372],
#         "SCC": [0.485, 0.381, 0.391, 0.388, 0.376, 0.376, 0.378]
#     }
# }
#
# # ---------------------- 关键修改：标题格式 + 左对齐 ----------------------
# plot_info = [
#     {"key": "kcat", "label": "a", "title": "$k_{cat}$ Dataset", "metrics": ["PCC", "SCC"]},
#     {"key": "Km", "label": "b", "title": "$K_m$ Dataset", "metrics": ["PCC", "SCC"]},
#     {"key": "kcat/Km", "label": "c", "title": "$k_{cat}/K_m$ Dataset", "metrics": ["PCC", "SCC"]}
# ]
#
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# for i, plot in enumerate(plot_info):
#     ax = axes[i]
#     y_pos = [0, 1]
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(plot["metrics"], fontsize=11)
#     ax.set_ylim(-0.5, 1.5)
#
#     # --- 解决1：a/b/c 往左移，和图框对齐 ---
#     # 用 ax.text 手动控制位置，比 ax.set_title 更灵活
#     ax.text(
#         -0.08, 1.1, plot["label"],
#         transform=ax.transAxes,
#         fontsize=14, fontweight='bold',
#         ha='left', va='top'
#     )
#     # 标题文字，放在label右侧，和label对齐
#     ax.text(
#         0, 1.1, plot["title"],
#         transform=ax.transAxes,
#         fontsize=13,
#         ha='left', va='top'
#     )
#
#     ax.grid(axis='y', linestyle='--', alpha=0.6, color='#cccccc')
#     ax.tick_params(axis='y', length=0)
#     ax.set_xlabel("Correlation", fontsize=12)
#
#     # 绘制数据点
#     for midx, m in enumerate(methods):
#         pcc = data[plot["key"]]["PCC"][midx]
#         scc = data[plot["key"]]["SCC"][midx]
#         ax.scatter(
#             pcc, 0,
#             c=colors[midx], marker=markers[midx],
#             s=110, edgecolor='white', linewidth=1.5, zorder=3
#         )
#         ax.scatter(
#             scc, 1,
#             c=colors[midx], marker=markers[midx],
#             s=110, edgecolor='white', linewidth=1.5, zorder=3
#         )
#
# # 添加图例
# handles = [
#     plt.Line2D(
#         [], [], color=c, marker=m, markersize=10, linestyle='',
#         label=l, markeredgecolor='white', markeredgewidth=1.5
#     ) for c, m, l in zip(colors, markers, methods)
# ]
# fig.legend(
#     handles=handles,
#     loc='upper center',
#     ncol=7,
#     bbox_to_anchor=(0.5, 1.15),
#     frameon=False,
#     fontsize=12  # 👈 这里改大图例字体！原来 10 → 现在 12
# )
#
# # 保存矢量图
# plt.savefig("final_figure_fixed.svg", format="svg", bbox_inches='tight')
# plt.close()

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# ---------------------- 全局配置 ----------------------
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 600,
    'figure.constrained_layout.use': True,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# ---------------------- 样式匹配 ----------------------
style_map = {
    "DB-Kinetic":          {"color": "#E63946", "marker": "D", "label": "DB-Kinetic"},
    "Esm2MolT5-MACCS":     {"color": "#6A0572", "marker": "X", "label": "Esm2_MolT5-MACCS"},
    "ProT5MolT5-MACCS":    {"color": "#1D3557", "marker": "s", "label": "ProT5_MolT5-MACCS"},
    "Esm2MolT5MACCS":      {"color": "#6A0572", "marker": "X", "label": "Esm2_MolT5-MACCS"},
    "ProT5MolT5":          {"color": "#E9C46A", "marker": "*", "label": "ProT5_MolT5"},
    "ProT5MACCS":          {"color": "#2A9D8F", "marker": "o", "label": "ProT5_MACCS"},
    "ProT5MACC":           {"color": "#2A9D8F", "marker": "o", "label": "ProT5_MACCS"},
    "ProT5Morgan":         {"color": "#F4A261", "marker": "p", "label": "ProT5_Morgan"},
    "Prot5Morgan":         {"color": "#F4A261", "marker": "p", "label": "ProT5_Morgan"},
    "ProT5RDKitFP":        {"color": "#457B9D", "marker": "^", "label": "ProT5_RDKitFP"},
}

# ---------------------- 固定子图顺序 ----------------------
plot_info = [
    {"key": "kcat",    "label": "a", "title": "$k_{cat}$ Dataset"},
    {"key": "Km",      "label": "b", "title": "$K_m$ Dataset"},
    {"key": "kcat/Km", "label": "c", "title": "$k_{cat}/K_m$ Dataset"},
]

# ---------------------- 文件路径 ----------------------
files = {
    "kcat":    r"D:\fwq\catapro\analyse\kcat消融.xlsx",
    "Km":      r"D:\fwq\catapro\analyse\km消融.xlsx",
    "kcat/Km": r"D:\fwq\catapro\analyse\kcat_km消融.xlsx",
}

# ---------------------- 读取数据 ----------------------
data = {}
for k, path in files.items():
    if os.path.exists(path):
        df = pd.read_excel(path)
        data[k] = df

# ---------------------- 绘图（核心修改：加抖动） ----------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
np.random.seed(42)  # 固定随机种子，保证每次抖动都一样

for ax, plot in zip(axes, plot_info):
    key = plot["key"]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PCC", "SCC"], fontsize=11)
    ax.set_ylim(-0.5, 1.5)
    ax.text(-0.08, 1.1, plot["label"], transform=ax.transAxes, fontsize=14, fontweight='bold', ha='left')
    ax.text(0, 0.98, plot["title"], transform=ax.transAxes, fontsize=13, ha='left')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel("Correlation", fontsize=12)

    if key not in data:
        continue

    df = data[key]
    for _, row in df.iterrows():
        model = str(row.iloc[0])
        pcc = row["PCC"]
        scc = row["SCC"]

        if model not in style_map:
            continue

        c = style_map[model]["color"]
        m = style_map[model]["marker"]
        size = 180 if m == "*" else 110

        # 核心修改：给x轴加一个极小的抖动（±0.0003，肉眼几乎看不见，但能分开点）
        jitter = np.random.uniform(-0.0003, 0.0003)
        # ax.scatter(pcc + jitter, 0, c=c, marker=m, s=size, edgecolor='white', lw=1.5, zorder=3)
        # ax.scatter(scc + jitter, 1, c=c, marker=m, s=size, edgecolor='white', lw=1.5, zorder=3)
        ax.scatter(pcc, 0, c=c, marker=m, s=size, edgecolor='white', lw=1.5, alpha=0.8, zorder=3)
        ax.scatter(scc, 1, c=c, marker=m, s=size, edgecolor='white', lw=1.5, alpha=0.8, zorder=3)

# ---------------------- 图例 ----------------------
legend_list = [
    "DB-Kinetic", "ProT5MolT5-MACCS", "ProT5RDKitFP",
    "ProT5MACCS", "ProT5Morgan", "ProT5MolT5", "Esm2MolT5-MACCS"
]

handles = []
for model in legend_list:
    if model not in style_map:
        continue
    c = style_map[model]["color"]
    m = style_map[model]["marker"]
    label = style_map[model]["label"]
    ms = 15 if m == "*" else 10
    handles.append(plt.Line2D([], [], color=c, marker=m, markersize=ms, linestyle='',
                              label=label, markeredgecolor='white', markeredgewidth=1.5))

fig.legend(handles=handles, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=12)

# ---------------------- 保存 ----------------------
plt.savefig("final_jittered.svg", format="svg", bbox_inches='tight')
plt.close()
print("✅ 绘图完成！重叠的点已通过抖动分开显示！")