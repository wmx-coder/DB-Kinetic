# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr
# import re
# import warnings
# warnings.filterwarnings('ignore')
#
# # ===================== 全局样式 =====================
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['axes.linewidth'] = 1.2
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['svg.fonttype'] = 'path'
#
# # ===================== 读取数据 =====================
# data_path = '/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/kcat_experiment2_processed.pkl'
# df = pd.read_pickle(data_path)
# df_mut = df[~df['is_wild_type']].copy()
#
# # ===================== 图a数据：误差箱线图 =====================
# mutant_df = df_mut.copy()
# mutant_df['abs_error'] = np.abs(mutant_df['log10_kcat'] - mutant_df['pred_log10_kcat'])
# group_order = ['normal', 'high', 'extreme']
# colors_box = ['#66c2a5', '#fc8d62', '#e78ac3']
#
# # ===================== 图b数据：fold区间PCC/SCC =====================
# df_mut['log10_fold'] = np.log10(df_mut['kcat_fold_vs_wt'].clip(lower=1e-8))
# bins = [-6, -4, -2, 0, 2, 4]
# bin_labels = [
#     '[-6,-4)\n(fold≈1e-6~1e-4)',
#     '[-4,-2)\n(fold≈1e-4~0.01)',
#     '[-2,0)\n(fold≈0.01~1)',
#     '(0,2]\n(fold≈1~100)',
#     '(2,4]\n(fold≈100~10000)'
# ]
# df_mut['fold_bin'] = pd.cut(df_mut['log10_fold'], bins=bins, labels=bin_labels, right=False)
#
# metrics = []
# for bl in bin_labels:
#     bd = df_mut[df_mut['fold_bin'] == bl]
#     if len(bd) < 2:
#         p, s = 0, 0
#     else:
#         p, _ = pearsonr(bd['log10_kcat'], bd['pred_log10_kcat'])
#         s, _ = spearmanr(bd['log10_kcat'], bd['pred_log10_kcat'])
#     metrics.append({'bin': bl, 'PCC': round(p, 2), 'SCC': round(s, 2), 'cnt': len(bd)})
# metrics_df = pd.DataFrame(metrics)
#
# # ===================== 图c数据：突变位点数 =====================
# def count_mutation_sites(s):
#     if pd.isna(s) or s.lower() == 'wild':
#         return 0
#     cnt = len(re.findall(r'[A-Z]\d+[A-Z]', str(s)))
#     return max(cnt, 1)
#
# df_mut['n_sites'] = df_mut['EnzymeType'].apply(count_mutation_sites)
# df_use = df_mut[df_mut['n_sites'].between(1, 6)]
# stat_list = []
# for n in range(1,7):
#     sub = df_use[df_use['n_sites'] == n]
#     if len(sub) >= 2:
#         p, _ = pearsonr(sub['log10_kcat'], sub['pred_log10_kcat'])
#         s, _ = spearmanr(sub['log10_kcat'], sub['pred_log10_kcat'])
#     else:
#         p, s = 0, 0
#     stat_list.append({'n': n, 'PCC': round(p, 2), 'SCC': round(s, 2)})
# stat_df = pd.DataFrame(stat_list)
#
# # ===================== 1×3 整合绘图 =====================
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
# fig.subplots_adjust(wspace=0.3)
#
# # -------------------- a 误差箱线图 --------------------
# sns.boxplot(
#     x='fold_group', y='abs_error', data=mutant_df, order=group_order,
#     palette=colors_box, width=0.2, fliersize=0, ax=ax1,
#     boxprops={'edgecolor': 'black', 'linewidth': 1.5},
#     whiskerprops={'color': 'black', 'linewidth': 1.2},
#     capprops={'color': 'black', 'linewidth': 1.2},
#     medianprops={'color': 'white', 'linewidth': 1.5}
# )
# ax1.set_ylim(-0.2, 4)
# ax1.set_title('Prediction Error Distribution by kcat Fold Change Group', fontweight='bold', fontsize=14)
# ax1.set_xlabel('kcat Change Group')
# ax1.set_ylabel('Absolute Error (|log10_kcat - pred_log10_kcat|)')
# ax1.grid(axis='y', alpha=0.3)
# for sp in ax1.spines.values():
#     sp.set_visible(True)
#     sp.set_linewidth(1.2)
#     sp.set_color('black')
# ax1.text(-0.05, 1.05, 'a', transform=ax1.transAxes, fontsize=18, fontweight='bold')
#
# # -------------------- b fold区间柱状图 --------------------
# x = np.arange(len(metrics_df))
# width = 0.35
# ax2.bar(x - width/2, metrics_df['PCC'], width, label='PCC', color='#66c2a5')
# ax2.bar(x + width/2, metrics_df['SCC'], width, label='SCC', color='#fc8d62')
# ax2.set_xticks(x)
# ax2.set_xticklabels(metrics_df['bin'], fontsize=9)
# ax2.set_ylim(0, 0.9)
# ax2.set_title('Model Performance Across Continuous fold Change', fontweight='bold', fontsize=14)
# ax2.set_ylabel('Correlation Coefficient')
# ax2.legend(fontsize=11)
# ax2.grid(axis='y', alpha=0.3)
# ax2.text(-0.05, 1.05, 'b', transform=ax2.transAxes, fontsize=18, fontweight='bold')
#
# # -------------------- c 突变位点数柱状图 --------------------
# x3 = np.arange(1, 7)
# ax3.bar(x3 - width/2, stat_df['PCC'], width, color='#66c2a5', label='PCC')
# ax3.bar(x3 + width/2, stat_df['SCC'], width, color='#fc8d62', label='SCC')
# ax3.set_xticks(x3)
# ax3.set_xticklabels([f'{i} site' for i in x3])
# ax3.set_ylim(0, 0.7)
# ax3.set_title('PCC/SCC by Mutation Site Count', fontweight='bold', fontsize=14)
# ax3.set_xlabel('Number of mutation sites')
# ax3.set_ylabel('Correlation Coefficient')
# ax3.legend(fontsize=11)
# ax3.grid(axis='y', alpha=0.3)
# ax3.text(-0.05, 1.05, 'c', transform=ax3.transAxes, fontsize=18, fontweight='bold')
#
# # ===================== 保存 =====================
# plt.tight_layout()
# #plt.savefig('/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/combined_3panel_abc.png', bbox_inches='tight', facecolor='white')
# plt.savefig('/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/combined_3panel_abc.svg', bbox_inches='tight')
# plt.close()
#
# print("✅ 三图整合完成：a b c 无括号 + 每个子图均有标题")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr
# import re
# import warnings
# warnings.filterwarnings('ignore')
#
# # ===================== 全局样式 =====================
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['axes.linewidth'] = 1.2
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['svg.fonttype'] = 'path'
#
# # ===================== 读取数据 =====================
# data_path = '/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/kcat_experiment2_processed.pkl'
# df = pd.read_pickle(data_path)
# df_mut = df[~df['is_wild_type']].copy()
#
# # ===================== 图a数据：误差箱线图 =====================
# mutant_df = df_mut.copy()
# mutant_df['abs_error'] = np.abs(mutant_df['log10_kcat'] - mutant_df['pred_log10_kcat'])
# group_order = ['normal', 'high', 'extreme']
# colors_box = ['#66c2a5', '#fc8d62', '#e78ac3']
#
# # ===================== 图b数据：fold区间PCC/SCC =====================
# df_mut['log10_fold'] = np.log10(df_mut['kcat_fold_vs_wt'].clip(lower=1e-8))
# bins = [-6, -4, -2, 0, 2, 4]
# bin_labels = [
#     '[-6,-4)\n(fold≈1e-6~1e-4)',
#     '[-4,-2)\n(fold≈1e-4~0.01)',
#     '[-2,0)\n(fold≈0.01~1)',
#     '(0,2]\n(fold≈1~100)',
#     '(2,4]\n(fold≈100~10000)'
# ]
# df_mut['fold_bin'] = pd.cut(df_mut['log10_fold'], bins=bins, labels=bin_labels, right=False)
#
# metrics = []
# for bl in bin_labels:
#     bd = df_mut[df_mut['fold_bin'] == bl]
#     if len(bd) < 2:
#         p, s = 0, 0
#     else:
#         p, _ = pearsonr(bd['log10_kcat'], bd['pred_log10_kcat'])
#         s, _ = spearmanr(bd['log10_kcat'], bd['pred_log10_kcat'])
#     metrics.append({'bin': bl, 'PCC': round(p, 2), 'SCC': round(s, 2), 'cnt': len(bd)})
# metrics_df = pd.DataFrame(metrics)
#
# # ===================== 图c数据：突变位点数 =====================
# def count_mutation_sites(s):
#     if pd.isna(s) or s.lower() == 'wild':
#         return 0
#     cnt = len(re.findall(r'[A-Z]\d+[A-Z]', str(s)))
#     return max(cnt, 1)
#
# df_mut['n_sites'] = df_mut['EnzymeType'].apply(count_mutation_sites)
# df_use = df_mut[df_mut['n_sites'].between(1, 6)]
# stat_list = []
# for n in range(1,7):
#     sub = df_use[df_use['n_sites'] == n]
#     if len(sub) >= 2:
#         p, _ = pearsonr(sub['log10_kcat'], sub['pred_log10_kcat'])
#         s, _ = spearmanr(sub['log10_kcat'], sub['pred_log10_kcat'])
#     else:
#         p, s = 0, 0
#     stat_list.append({'n': n, 'PCC': round(p, 2), 'SCC': round(s, 2)})
# stat_df = pd.DataFrame(stat_list)
#
# # ===================== 1×3 整合绘图 =====================
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
# fig.subplots_adjust(wspace=0.3)
#
# # -------------------- a 误差箱线图 + 白色中位数 --------------------
# sns.boxplot(
#     x='fold_group', y='abs_error', data=mutant_df, order=group_order,
#     palette=colors_box, width=0.2, fliersize=0, ax=ax1,
#     boxprops={'edgecolor': 'black', 'linewidth': 1.5},
#     whiskerprops={'color': 'black', 'linewidth': 1.2},
#     capprops={'color': 'black', 'linewidth': 1.2},
#     medianprops={'color': 'white', 'linewidth': 1.5}
# )
# ax1.set_ylim(-0.2, 4)
# ax1.set_title('Prediction Error Distribution by kcat Fold Change Group', fontweight='bold', fontsize=14)
# ax1.set_xlabel('kcat Change Group')
# ax1.set_ylabel('Absolute Error (|log10_kcat - pred_log10_kcat|)')
# ax1.grid(axis='y', alpha=0.3)
#
# for i, g in enumerate(group_order):
#     med = mutant_df[mutant_df['fold_group']==g]['abs_error'].median()
#     ax1.text(i, med + 0.08, f'{med:.2f}', ha='center', fontsize=11, fontweight='bold', color='white')
#
# ax1.text(-0.05, 1.05, 'a', transform=ax1.transAxes, fontsize=18, fontweight='bold')
#
# # -------------------- b 图：彻底解决重叠！上下+左右双重错开 --------------------
# x = np.arange(len(metrics_df))
# width = 0.35
#
# bars_pcc = ax2.bar(x - width/2, metrics_df['PCC'], width, label='PCC', color='#66c2a5')
# bars_scc = ax2.bar(x + width/2, metrics_df['SCC'], width, label='SCC', color='#fc8d62')
#
# ax2.set_xticks(x)
# ax2.set_xticklabels(metrics_df['bin'], fontsize=9)
# ax2.set_ylim(0, 1.05)
# ax2.set_title('Model Performance Across Continuous fold Change', fontweight='bold', fontsize=14)
# ax2.set_ylabel('Correlation Coefficient')
# ax2.legend(fontsize=11, loc='upper left')
# ax2.grid(axis='y', alpha=0.3)
#
# # ========== 核心修复：上下+左右双重错开，永不重叠 ==========
# for i, bar in enumerate(bars_pcc):
#     height = bar.get_height()
#     cnt = metrics_df.iloc[i]['cnt']
#     # PCC标注：偏左，垂直位置稍高
#     ax2.text(bar.get_x() + bar.get_width()/2 - 0.04, height + 0.04,
#              f"{height:.2f}\n(n={cnt})", ha='center', va='bottom', fontsize=10)
#
# for i, bar in enumerate(bars_scc):
#     height = bar.get_height()
#     cnt = metrics_df.iloc[i]['cnt']
#     # SCC标注：偏右，垂直位置稍低
#     ax2.text(bar.get_x() + bar.get_width()/2 + 0.04, height + 0.01,
#              f"{height:.2f}\n(n={cnt})", ha='center', va='bottom', fontsize=10)
#
# ax2.text(-0.05, 1.05, 'b', transform=ax2.transAxes, fontsize=18, fontweight='bold')
#
# # -------------------- c 突变位点数 --------------------
# x3 = np.arange(1, 7)
# ax3.bar(x3 - width/2, stat_df['PCC'], width, color='#66c2a5', label='PCC')
# ax3.bar(x3 + width/2, stat_df['SCC'], width, color='#fc8d62', label='SCC')
# ax3.set_xticks(x3)
# ax3.set_xticklabels([f'{i} site' for i in x3])
# ax3.set_ylim(0, 0.7)
# ax3.set_title('PCC/SCC by Mutation Site Count', fontweight='bold', fontsize=14)
# ax3.set_xlabel('Number of mutation sites')
# ax3.set_ylabel('Correlation Coefficient')
# ax3.legend(fontsize=11)
# ax3.grid(axis='y', alpha=0.3)
#
# for i, row in stat_df.iterrows():
#     ax3.text(x3[i]-width/2, row['PCC']+0.02, f"{row['PCC']:.2f}", ha='center', fontsize=10)
#     ax3.text(x3[i]+width/2, row['SCC']+0.02, f"{row['SCC']:.2f}", ha='center', fontsize=10)
#
# ax3.text(-0.05, 1.05, 'c', transform=ax3.transAxes, fontsize=18, fontweight='bold')
#
# # ===================== 保存 =====================
# plt.tight_layout()
# plt.savefig('/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/combined_3panel_abc.svg', bbox_inches='tight')
# plt.close()
#
# print("✅ 完美！b图标注上下+左右双重错开，彻底不重叠！")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局样式 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'path'

# ===================== 读取数据 =====================
data_path = '/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/kcat_experiment2_processed.pkl'
df = pd.read_pickle(data_path)
df_mut = df[~df['is_wild_type']].copy()

# ===================== 图a数据：误差箱线图 =====================
mutant_df = df_mut.copy()
mutant_df['abs_error'] = np.abs(mutant_df['log10_kcat'] - mutant_df['pred_log10_kcat'])
group_order = ['normal', 'high', 'extreme']
colors_box = ['#66c2a5', '#fc8d62', '#e78ac3']

# ===================== 图b数据：fold区间PCC/SCC =====================
df_mut['log10_fold'] = np.log10(df_mut['kcat_fold_vs_wt'].clip(lower=1e-8))
bins = [-6, -4, -2, 0, 2, 4]
bin_labels = [
    '[-6,-4)\n(fold≈1e-6~1e-4)',
    '[-4,-2)\n(fold≈1e-4~0.01)',
    '[-2,0)\n(fold≈0.01~1)',
    '(0,2]\n(fold≈1~100)',
    '(2,4]\n(fold≈100~10000)'
]
df_mut['fold_bin'] = pd.cut(df_mut['log10_fold'], bins=bins, labels=bin_labels, right=False)

metrics = []
for bl in bin_labels:
    bd = df_mut[df_mut['fold_bin'] == bl]
    if len(bd) < 2:
        p, s = 0, 0
    else:
        p, _ = pearsonr(bd['log10_kcat'], bd['pred_log10_kcat'])
        s, _ = spearmanr(bd['log10_kcat'], bd['pred_log10_kcat'])
    metrics.append({'bin': bl, 'PCC': round(p, 2), 'SCC': round(s, 2), 'cnt': len(bd)})
metrics_df = pd.DataFrame(metrics)

# ===================== 图c数据：突变位点数 =====================
def count_mutation_sites(s):
    if pd.isna(s) or s.lower() == 'wild':
        return 0
    cnt = len(re.findall(r'[A-Z]\d+[A-Z]', str(s)))
    return max(cnt, 1)

df_mut['n_sites'] = df_mut['EnzymeType'].apply(count_mutation_sites)
df_use = df_mut[df_mut['n_sites'].between(1, 6)]
stat_list = []
for n in range(1,7):
    sub = df_use[df_use['n_sites'] == n]
    if len(sub) >= 2:
        p, _ = pearsonr(sub['log10_kcat'], sub['pred_log10_kcat'])
        s, _ = spearmanr(sub['log10_kcat'], sub['pred_log10_kcat'])
    else:
        p, s = 0, 0
    stat_list.append({'n': n, 'PCC': round(p, 2), 'SCC': round(s, 2)})
stat_df = pd.DataFrame(stat_list)

# ===================== 1×3 整合绘图 =====================
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
fig.subplots_adjust(wspace=0.3)

# -------------------- a 误差箱线图 + 白色中位数 --------------------
sns.boxplot(
    x='fold_group', y='abs_error', data=mutant_df, order=group_order,
    palette=colors_box, width=0.2, fliersize=0, ax=ax1,
    boxprops={'edgecolor': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1.2},
    capprops={'color': 'black', 'linewidth': 1.2},
    medianprops={'color': 'white', 'linewidth': 1.5}
)
ax1.set_ylim(-0.2, 4)
# 修改标题：kcat 斜体下标
ax1.set_title(r'Prediction Error Distribution by $\mathit{k}_\mathrm{cat}$ Fold Change Group', fontweight='bold', fontsize=14)
ax1.set_xlabel('$\mathit{k}_\mathrm{cat}$ Change Group', fontsize=12)
ax1.set_ylabel('Absolute Error (|log10_kcat - pred_log10_kcat|)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', labelsize=12)  # 横轴放大
ax1.tick_params(axis='y', labelsize=12)

for i, g in enumerate(group_order):
    med = mutant_df[mutant_df['fold_group']==g]['abs_error'].median()
    ax1.text(i, med + 0.08, f'{med:.2f}', ha='center', fontsize=11, fontweight='bold', color='white')

# ========== abc 往左移动 ==========
ax1.text(-0.12, 1.05, 'a', transform=ax1.transAxes, fontsize=18, fontweight='bold')

# # -------------------- b 图：优化数字 + 去掉n + 横轴放大 ==========
# x = np.arange(len(metrics_df))
# width = 0.35
#
# bars_pcc = ax2.bar(x - width/2, metrics_df['PCC'], width, label='PCC', color='#66c2a5')
# bars_scc = ax2.bar(x + width/2, metrics_df['SCC'], width, label='SCC', color='#fc8d62')
#
# ax2.set_xticks(x)
# # 横轴放大 + 清晰
# ax2.set_xticklabels(metrics_df['bin'], fontsize=11)
# ax2.set_ylim(0, 1.05)
# # 标题斜体
# ax2.set_title(r'Model Performance Across Continuous fold Change ($\mathit{k}_\mathrm{cat}$)', fontweight='bold', fontsize=14)
# ax2.set_ylabel('Correlation Coefficient', fontsize=12)
# ax2.legend(fontsize=11, loc='upper left')
# ax2.grid(axis='y', alpha=0.3)
# ax2.tick_params(axis='y', labelsize=12)
#
# # ========== 只显示数值，大小和C图一致 ==========
# for i, bar in enumerate(bars_pcc):
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
#              f"{height:.2f}", ha='center', va='bottom', fontsize=10)  # 和C图一样
#
# for i, bar in enumerate(bars_scc):
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
#              f"{height:.2f}", ha='center', va='bottom', fontsize=10)
#
# # ========== n 数量 移到 横轴 最底部 ==========
# for i, row in metrics_df.iterrows():
#     ax2.text(i, -0.15, f"n={row['cnt']}", ha='center', fontsize=10, fontweight='bold')
#
# ax2.text(-0.12, 1.05, 'b', transform=ax2.transAxes, fontsize=18, fontweight='bold')

# -------------------- b 图（去掉 fold 文字，n 调整到横轴下方） --------------------
bin_labels_clean = [
    '[-6,-4)',
    '[-4,-2)',
    '[-2,0)',
    '(0,2]',
    '(2,4]'
]

x = np.arange(len(bin_labels_clean))
width = 0.35

bars_pcc = ax2.bar(x - width/2, metrics_df['PCC'], width, label='PCC', color='#66c2a5')
bars_scc = ax2.bar(x + width/2, metrics_df['SCC'], width, label='SCC', color='#fc8d62')

ax2.set_xticks(x)
ax2.set_xticklabels(bin_labels_clean, fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.set_title(r'Model Performance Across Continuous fold Change ($\mathit{k}_\mathrm{cat}$)', fontweight='bold', fontsize=14)
ax2.set_ylabel('Correlation Coefficient', fontsize=12)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='y', labelsize=12)

# 只显示数值
for i, bar in enumerate(bars_pcc):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f"{height:.2f}", ha='center', va='bottom', fontsize=10)

for i, bar in enumerate(bars_scc):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f"{height:.2f}", ha='center', va='bottom', fontsize=10)

# ========== 关键修改：n 位置和字体 ==========
for i, row in metrics_df.iterrows():
    ax2.text(i, -0.10, f"n={row['cnt']}",
             ha='center', va='top', fontsize=11)  # 与横轴刻度字体一致

ax2.text(-0.12, 1.05, 'b', transform=ax2.transAxes, fontsize=18, fontweight='bold')

# -------------------- c 突变位点数 --------------------
x3 = np.arange(1, 7)
ax3.bar(x3 - width/2, stat_df['PCC'], width, color='#66c2a5', label='PCC')
ax3.bar(x3 + width/2, stat_df['SCC'], width, color='#fc8d62', label='SCC')
ax3.set_xticks(x3)
ax3.set_xticklabels([f'{i} site' for i in x3], fontsize=12)
ax3.set_ylim(0, 0.7)
ax3.set_title('PCC/SCC by Mutation Site Count', fontweight='bold', fontsize=14)
ax3.set_xlabel('Number of mutation sites', fontsize=12)
ax3.set_ylabel('Correlation Coefficient', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)
ax3.tick_params(axis='y', labelsize=12)

for i, row in stat_df.iterrows():
    ax3.text(x3[i]-width/2, row['PCC']+0.02, f"{row['PCC']:.2f}", ha='center', fontsize=10)
    ax3.text(x3[i]+width/2, row['SCC']+0.02, f"{row['SCC']:.2f}", ha='center', fontsize=10)

ax3.text(-0.12, 1.05, 'c', transform=ax3.transAxes, fontsize=18, fontweight='bold')

# ===================== 保存 =====================
plt.tight_layout()
plt.savefig('/mnt/usb1/wmx/catapro/analyse/eitlem/kcat/combined_3panel_abc.svg', bbox_inches='tight')
plt.close()

print("✅ 全部修改完成！abc左移 + 横轴放大 + b图优化 + 斜体kcat")