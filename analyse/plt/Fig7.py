# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# import matplotlib
#
# matplotlib.use('Agg')
#
# # ===================== 1. 路径配置 =====================
# base_dir = r"D:\fwq\catapro\analyse\eitlem\case"
#
# files = {
#     "kcat": {
#         "path": fr"{base_dir}\kcat\case\kcat.csv",
#         "exp_col": "log10_kcat",
#         "pred_col": "pred_log10_kcat",
#         "label": "a",
#         "title": "kcat - 16 Groups"
#     },
#     "kcat_Km": {
#         "path": fr"{base_dir}\kcat_km\case\kcat_km.csv",
#         "exp_col": "log10_kcat_over_Km",
#         "pred_col": "pred_fusion_log10[kcat/Km]",
#         "label": "b",
#         "title": "kcat/Km - 16 Groups"
#     },
#     "Km": {
#         "path": fr"{base_dir}\km\case\km.csv",
#         "exp_col": "log10_Km",
#         "pred_col": "pred_log10[Km(mM)]",
#         "label": "c",
#         "title": "Km - 16 Groups"
#     }
# }
#
# # ===================== 2. 全局样式 =====================
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['svg.fonttype'] = 'path'
# plt.rcParams['axes.linewidth'] = 1.2
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['grid.alpha'] = 0.3
#
#
# # ===================== 3. 核心绘图（精准修复所有问题） =====================
# def plot_combined_figure():
#     # 关键修复1：调整画布尺寸和行间距，解决ab/bc间距过大
#     fig = plt.figure(figsize=(24, 48), facecolor='white')
#     # 大幅压缩行间距(hspace)，让3行紧凑对齐
#     gs = fig.add_gridspec(3, 1, hspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
#
#     # 关键修复2：统一3行标题的y坐标，解决a标题错位、c标题太远
#     title_y_pos = [0.97, 0.64, 0.31]  # 3行标题的精准y位置，完全对齐
#     label_y_pos = [0.97, 0.64, 0.31]  # a/b/c标注和标题同高，对齐
#
#     for row_idx, (name, cfg) in enumerate(files.items()):
#         print(f"📊 正在处理 {cfg['title'].split(' - ')[0]} ...")
#
#         # 读取数据（完全沿用原始逻辑）
#         df = pd.read_csv(cfg["path"])
#         df = df[~df["EnzymeType"].str.contains(r"wild|WT|Wild", case=False, na=False)]
#         df["group"] = df["UniProtID"].astype(str) + " | " + df["Substrate"].astype(str)
#         groups = df["group"].unique()[:16]
#
#         # 关键修复3：调整4×4子图内部间距，让子图更紧凑
#         gs_row = gs[row_idx].subgridspec(4, 4, wspace=0.28, hspace=0.38)
#         axes = gs_row.subplots().flatten()
#
#         # 逐组绘图（完全复刻原始样式）
#         for i, g in enumerate(groups):
#             ax = axes[i]
#             sub = df[df["group"] == g].copy().sort_values(cfg["exp_col"])
#             if len(sub) < 2:
#                 ax.set_visible(False)
#                 continue
#
#             x_labels = sub["EnzymeType"].values
#             y_exp = sub[cfg["exp_col"]].values
#             y_pred = sub[cfg["pred_col"]].values
#
#             pcc, _ = pearsonr(y_exp, y_pred)
#             pcc_str = f"PCC={pcc:.2f}"
#
#             ax.plot(x_labels, y_exp, 'o-', label='Exp', color='#2E86AB', linewidth=2.5, markersize=5)
#             ax.plot(x_labels, y_pred, 's-', label='Pred', color='#F24236', linewidth=2.5, markersize=5)
#
#             ax.set_title(f"{g}\n{pcc_str}", fontsize=10, pad=6)
#             ax.tick_params(axis='x', rotation=60, labelsize=7)
#             ax.legend(fontsize=8, loc='upper right')
#             ax.grid(True, alpha=0.3)
#
#             y_min = min(y_exp.min(), y_pred.min()) - 0.2
#             y_max = max(y_exp.max(), y_pred.max()) + 0.2
#             ax.set_ylim(y_min, y_max)
#
#         # 隐藏多余子图
#         for j in range(len(groups), 16):
#             axes[j].set_visible(False)
#
#         # 关键修复4：精准定位标题，3行完全对齐
#         fig.text(
#             0.5, title_y_pos[row_idx],
#             cfg["title"],
#             fontsize=16,
#             fontweight='bold',
#             ha='center',
#             transform=fig.transFigure
#         )
#
#         # 关键修复5：a/b/c标注和标题同高，完全对齐
#         fig.text(
#             0.02, label_y_pos[row_idx],
#             cfg["label"],
#             fontsize=20,
#             fontweight='bold',
#             transform=fig.transFigure
#         )
#
#     # ===================== 4. 保存SVG =====================
#     save_path = fr"{base_dir}\combined_3panel_abc_final_fixed.svg"
#     plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300, facecolor='white')
#     plt.close()
#     print(f"✅ 所有问题彻底修复！SVG已保存至：\n{save_path}")
#
#
# if __name__ == "__main__":
#     plot_combined_figure()
#     print("\n🎉 3行1列大图生成成功，标题、间距100%对齐！")

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib

matplotlib.use('Agg')

# ===================== 1. 路径配置 =====================
base_dir = r"D:\fwq\catapro\analyse\eitlem\fig6"

files = {
    "kcat": {
        "path": fr"{base_dir}\kcat\fig6\kcat.csv",
        "exp_col": "log10_kcat",
        "pred_col": "pred_log10_kcat",
        "label": "a",
        "title": r"$\mathit{k}_\mathrm{cat}$ - 4 Groups"
    },
    "kcat_Km": {
        "path": fr"{base_dir}\kcat_km\fig6\kcat_km.csv",
        "exp_col": "log10_kcat_over_Km",
        "pred_col": "pred_fusion_log10[kcat/Km]",
        "label": "b",
        "title": r"$\mathit{k}_\mathrm{cat}/\mathit{K}_\mathrm{m}$ - 4 Groups"
    },
    "Km": {
        "path": fr"{base_dir}\km\fig6\km.csv",
        "exp_col": "log10_Km",
        "pred_col": "pred_log10[Km(mM)]",
        "label": "c",
        "title": r"$\mathit{K}_\mathrm{m}$ - 4 Groups"
    }
}

# ===================== 2. 全局样式 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300


# ===================== 3. 核心绘图 =====================
def plot_combined_figure():
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    gs = fig.add_gridspec(3, 1, hspace=0.60, top=0.92, bottom=0.08, left=0.08, right=0.95)

    for row_idx, (name, cfg) in enumerate(files.items()):
        print(f"📊 正在处理 {cfg['title']} ...")

        df = pd.read_csv(cfg["path"])
        df = df[~df["EnzymeType"].str.contains(r"wild|WT|Wild", case=False, na=False)]
        df["group"] = df["UniProtID"].astype(str) + " | " + df["Substrate"].astype(str)
        groups = df["group"].unique()[:4]

        bbox = gs[row_idx].get_position(fig)
        current_top = bbox.y1

        gs_row = gs[row_idx].subgridspec(1, 4, wspace=0.25)
        axes = gs_row.subplots().flatten()

        for i, g in enumerate(groups):
            ax = axes[i]
            sub = df[df["group"] == g].copy().sort_values(cfg["exp_col"])

            if len(sub) < 2:
                ax.set_visible(False)
                continue

            x_labels = sub["EnzymeType"].values
            y_exp = sub[cfg["exp_col"]].values
            y_pred = sub[cfg["pred_col"]].values
            pcc, _ = pearsonr(y_exp, y_pred)

            ax.plot(x_labels, y_exp, 'o-', label='Exp', color='#2E86AB', linewidth=2.5, markersize=6)
            ax.plot(x_labels, y_pred, 's-', label='Pred', color='#F24236', linewidth=2.5, markersize=6)

            ax.set_title(f"{g}\nPCC={pcc:.2f}", fontsize=10, pad=8)

            # 横轴字体放大 + 超长自动两行换行
            wrapped_labels = []
            for lab in x_labels:
                if len(lab) > 8:
                    mid = len(lab) // 2
                    wrapped = lab[:mid] + "\n" + lab[mid:]
                else:
                    wrapped = lab
                wrapped_labels.append(wrapped)

            ax.set_xticks(range(len(wrapped_labels)))
            ax.set_xticklabels(wrapped_labels, fontsize=11)   # 字体再放大
            ax.tick_params(axis='y', labelsize=9)

            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.4)

            y_min = min(y_exp.min(), y_pred.min()) - 0.2
            y_max = max(y_exp.max(), y_pred.max()) + 0.2
            ax.set_ylim(y_min, y_max)

        for j in range(len(groups), 4):
            axes[j].set_visible(False)

        # 标题往上挪：+0.015 改成 +0.03
        fig.text(
            0.5, current_top + 0.03,
            cfg["title"],
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='bottom',
            transform=fig.transFigure
        )

        fig.text(
            0.02, current_top + 0.03,
            cfg["label"],
            fontsize=24,
            fontweight='bold',
            va='bottom',
            transform=fig.transFigure
        )

    # ===================== 4. 保存结果 =====================
    save_path = fr"{base_dir}\combined_1x4_3row_dynamic.svg"
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ 已完成：横轴字体放大、上方标题整体上移")


if __name__ == "__main__":
    plot_combined_figure()