import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== 配置 =====================
folders = [
    r"C:\Users\29446\Desktop\kcat",
    r"C:\Users\29446\Desktop\km",
    r"C:\Users\29446\Desktop\kcat_km"
]
metrics = ["PCC", "SCC", "Rmse"]
sub_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

# 模型固定颜色映射（所有数据集统一）
color_map = {
    "DB-Kinetic": "#94c0f4",    # 淡蓝
    "CataPro": "#ffc699",       # 淡橙
    "DLKcat": "#a1f0ac",        # 淡绿
    "EITLEM": "#ff9ca0",        # 淡红
    "EITLEM(w.o. TL)": "#c8b0ff",# 淡紫
    "UniKP": "#e6c9b0"          # 淡棕
}

# 文件夹名称转为斜体下标格式
def folder2latex(name):
    if name == "kcat":
        return r"$\mathit{k}_\text{cat}$"
    elif name == "km":
        return r"$\mathit{K}_\text{m}$"
    elif name == "kcat_km":
        return r"$\mathit{k}_\text{cat}$-$\mathit{K}_\text{m}$"
    else:
        return name

# ===================== 读取数据 =====================
def load_all_models(folder):
    model_data = {}
    for f in os.listdir(folder):
        if f.endswith(".xlsx"):
            model_name = f.replace(".xlsx", "")
            df = pd.read_excel(os.path.join(folder, f))
            model_data[model_name] = df.head(10).copy()
    return model_data

# 排序：DB-Kinetic 放最前
def get_order(model_data):
    ordered = []
    if "DB-Kinetic" in model_data:
        ordered.append("DB-Kinetic")
    for m in sorted(model_data.keys()):
        if m != "DB-Kinetic":
            ordered.append(m)
    ordered_colors = [color_map[m] for m in ordered]
    return ordered, ordered_colors

# ===================== 绘制 3×3 大图 =====================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 13

fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=250)
axes = axes.flatten()
idx = 0

for folder in folders:
    models = load_all_models(folder)
    ordered_models, ordered_colors = get_order(models)
    folder_latex = folder2latex(os.path.basename(folder))

    for metric in metrics:
        ax = axes[idx]
        means = []
        stds = []
        folds = []

        for m in ordered_models:
            vals = models[m][metric].values
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1))
            folds.append(vals)

        x = np.arange(len(ordered_models))
        width = 0.6

        # 柱状图+误差棒
        ax.bar(x, means, width,
               color=ordered_colors, edgecolor="black", linewidth=1.0,
               yerr=stds, capsize=5, error_kw={"ecolor": "black", "elinewidth": 1.2})

        # 散点：同色填充+白色外圈
        for i, (data, color) in enumerate(zip(folds, ordered_colors)):
            xj = np.random.uniform(x[i] - width/2 + 0.06, x[i] + width/2 - 0.06, size=len(data))
            ax.scatter(xj, data, s=60,
                       facecolor=color, edgecolor="white", linewidth=1.3, zorder=3)

        # 子图标号（和你箱型图样式一致）
        ax.set_title(sub_labels[idx], loc='left', fontsize=20, fontweight='bold', pad=10)

        # ===================== 关键修改1：横轴模型名倾斜30度 =====================
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_models, fontsize=11, rotation=30, ha='right')

        # 坐标轴样式
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylabel(metric, fontsize=15, fontweight='bold')

        # ===================== 关键修改2：所有标题加粗 =====================
        ax.set_title(f"{folder_latex} - {metric}", fontsize=15, fontweight='bold', pad=10)

        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(bottom=0)

        idx += 1

plt.tight_layout()
save_path = r"C:\Users\29446\Desktop\all_in_one_figure.svg"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ 已修改完成：横轴文字倾斜防重叠，所有标题已加粗！")