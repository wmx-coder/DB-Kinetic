# import pandas as pd
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from itertools import combinations
# from scikit_posthocs import posthoc_dunn
# import os
# import random
#
# # ======================================================================================
# #
# #   【1】kcat 部分代码（完全不变）
# #
# # ======================================================================================
# file_config = {
#     "N≥20": {
#         "my": "/mnt/usb1/wmx/catapro/analyse/kcat/kcat_pred_by_fold_20.pkl",
#         "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_kcat/N20.pkl",
#         "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/kcat/N20_1.pkl",
#         "UniKP": "/mnt/usb1/wmx/unikp/analysis/kcat/N20.pkl",
#         "Eitlem": "/mnt/usb1/wmx/eitlem/kcat/kcat_N20_final_pred.pkl"
#     }
# }
#
# def load_kcat_data(file_paths, n_threshold):
#     data_dict = {}
#     for model, path in file_paths.items():
#         print(f"\n=== 加载 {model} 数据（{n_threshold}）===")
#         if not os.path.exists(path):
#             print(f"❌ 错误：{model} 文件不存在 → {path}")
#             raise FileNotFoundError(f"Missing file for {model}: {path}")
#         try:
#             df = pd.read_pickle(path)
#             df = df[df['kcat(s^-1)'] > 0].reset_index(drop=True)
#             data_dict[model] = df
#             print(f"✅ {model} 数据加载完成，有效行数：{len(df)}")
#         except Exception as e:
#             print(f"❌ 读取 {model} 失败：{e}")
#             raise e
#     return data_dict
#
# def unify_log10_scale(data_dict, n_threshold):
#     unified_data = {}
#     df_my = data_dict["my"].copy()
#     df_my['true_log10_kcat'] = np.log10(df_my['kcat(s^-1)'])
#     df_my.rename(columns={'pred_log10_kcat': 'model_pred_log10'}, inplace=True)
#     unified_data["my"] = df_my
#
#     df_cata = data_dict["CataPro"].copy()
#     df_cata['true_log10_kcat'] = np.log10(df_cata['kcat(s^-1)'])
#     df_cata.rename(columns={'pred_log10[kcat(s^-1)]': 'model_pred_log10'}, inplace=True)
#     unified_data["CataPro"] = df_cata
#
#     df_dlk = data_dict["DLKcat"].copy()
#     df_dlk['true_log10_kcat'] = np.log10(df_dlk['kcat(s^-1)'])
#     if 'Pred_Kcat_log2' in df_dlk.columns:
#         df_dlk['model_pred_log10'] = df_dlk['Pred_Kcat_log2'] / np.log2(10)
#     else:
#         df_dlk['model_pred_log10'] = df_dlk.get('pred_log10_kcat', np.nan)
#     unified_data["DLKcat"] = df_dlk
#
#     df_uni = data_dict["UniKP"].copy()
#     df_uni['true_log10_kcat'] = np.log10(df_uni['kcat(s^-1)'])
#     df_uni.rename(columns={'pred_log10_kcat': 'model_pred_log10'}, inplace=True)
#     unified_data["UniKP"] = df_uni
#
#     df_eit = data_dict["Eitlem"].copy()
#     df_eit['true_log10_kcat'] = np.log10(df_eit['kcat(s^-1)'])
#     df_eit['model_pred_log10'] = df_eit['pred']
#     unified_data["Eitlem"] = df_eit
#
#     common_reactions = set(unified_data["my"]['reaction_id'])
#     for model in ["CataPro", "DLKcat", "UniKP", "Eitlem"]:
#         if model in unified_data and not unified_data[model].empty:
#             common_reactions = common_reactions & set(unified_data[model]['reaction_id'])
#     print(f"\n🔍 5个模型共有的reaction_id数量：{len(common_reactions)}")
#
#     for model in unified_data:
#         df = unified_data[model]
#         if df.empty:
#             unified_data[model] = df
#             print(f"⚠️ {model} 原始数据为空，跳过过滤")
#             continue
#         df = df[df['reaction_id'].isin(common_reactions)]
#         reaction_counts = df['reaction_id'].value_counts()
#         valid_reactions = reaction_counts[reaction_counts >= n_threshold].index
#         df = df[df['reaction_id'].isin(valid_reactions)]
#         unified_data[model] = df.reset_index(drop=True)
#         print(f"✅ {model} 数据过滤完成，有效reaction数：{len(valid_reactions)}，有效行数：{len(df)}")
#     return unified_data
#
# def calculate_metrics_per_reaction(unified_data):
#     metrics_dict = {}
#     for model, df in unified_data.items():
#         model_metrics = {}
#         print(f"\n🔢 计算 {model} 模型指标...")
#         if df.empty:
#             print(f"⚠️ {model} 数据为空，跳过指标计算")
#             metrics_dict[model] = model_metrics
#             continue
#         for idx, (reaction_id, group) in enumerate(df.groupby('reaction_id')):
#             true_vals = group['true_log10_kcat'].values
#             pred_vals = group['model_pred_log10'].values
#             if len(np.unique(true_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
#                 continue
#             scc, _ = stats.spearmanr(true_vals, pred_vals)
#             scc = np.nan_to_num(scc)
#             pcc, _ = stats.pearsonr(true_vals, pred_vals)
#             pcc = np.nan_to_num(pcc)
#             indices = list(range(len(group)))
#             combinations_list = list(combinations(indices, 2))
#             correct = 0
#             for i, j in combinations_list:
#                 true_i_higher = true_vals[i] > true_vals[j]
#                 pred_i_higher = pred_vals[i] > pred_vals[j]
#                 if true_i_higher == pred_i_higher:
#                     correct += 1
#             accuracy = correct / len(combinations_list) if len(combinations_list) > 0 else 0
#             model_metrics[reaction_id] = {'scc': scc, 'pcc': pcc, 'accuracy': accuracy}
#         metrics_dict[model] = model_metrics
#         print(f"✅ {model} 指标计算完成，有效reaction数：{len(model_metrics)}")
#     return metrics_dict
#
# def organize_metrics(metrics_dict):
#     scc_data = {}
#     pcc_data = {}
#     acc_data = {}
#     models = ["my", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     for model in models:
#         model_metrics = metrics_dict.get(model, {})
#         scc_data[model] = [v['scc'] for v in model_metrics.values()] if model_metrics else []
#         pcc_data[model] = [v['pcc'] for v in model_metrics.values()] if model_metrics else []
#         acc_data[model] = [v['accuracy'] for v in model_metrics.values()] if model_metrics else []
#         print(f"📊 {model} - SCC：{len(scc_data[model])}, PCC：{len(pcc_data[model])}, 准确率：{len(acc_data[model])}")
#     return scc_data, pcc_data, acc_data
#
# # ======================================================================================
# #
# #   【2】Km 部分代码（完全不变）
# #
# # ======================================================================================
# km_file_paths = {
#     "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_km/km20.pkl",
#     "my": "/mnt/usb1/wmx/catapro/analyse/km/km-data_filtered_N20_with_pred.pkl",
#     "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/km/km_N20_matched_result.pkl",
#     "UniKP": "/mnt/usb1/wmx/unikp/analysis/km/km_N20.pkl",
#     "Eitlem": "/mnt/usb1/wmx/eitlem/kcat/km_N20_final_pred.pkl"
# }
#
# def load_and_clean_km_data(file_paths):
#     data_dict = {}
#     for model, path in file_paths.items():
#         print(f"\n=== 加载 {model} 数据 ===")
#         df = pd.read_pickle(path)
#         df = df[(df['Km(M)'] > 0) & (df['reaction_id'].notnull())].reset_index(drop=True)
#         print(f"   清洗后总行数：{len(df)}")
#         data_dict[model] = df
#     return data_dict
#
# def unify_km_log10_scale(data_dict):
#     unified_data = {}
#     df_cata = data_dict["CataPro"].copy()
#     df_cata['true_log10_km'] = df_cata['log10_Km']
#     df_cata['model_pred_log10'] = df_cata['pred_log10[Km(mM)]']
#     unified_data["CataPro"] = df_cata
#
#     df_my = data_dict["my"].copy()
#     df_my['true_log10_km'] = df_my['log10_Km']
#     df_my['model_pred_log10'] = df_my['pred_log10[Km(mM)]']
#     unified_data["my"] = df_my
#
#     df_dlk = data_dict["DLKcat"].copy()
#     df_dlk['true_log10_km'] = df_dlk['log10_Km']
#     df_dlk['model_pred_log10'] = df_dlk['Pred_Km_log2'] / np.log2(10)
#     unified_data["DLKcat"] = df_dlk
#
#     df_uni = data_dict["UniKP"].copy()
#     df_uni['true_log10_km'] = df_uni['log10_Km']
#     df_uni['model_pred_log10'] = df_uni['pred_log10_km']
#     unified_data["UniKP"] = df_uni
#
#     df_eit = data_dict["Eitlem"].copy()
#     df_eit['true_log10_km'] = np.log10(df_eit['Km(M)'])
#     df_eit['model_pred_log10'] = df_eit['pred']
#     unified_data["Eitlem"] = df_eit
#
#     common_reactions = set(unified_data["my"]['reaction_id'])
#     for model in ["CataPro", "DLKcat", "UniKP", "Eitlem"]:
#         common_reactions = common_reactions & set(unified_data[model]['reaction_id'])
#     print(f"\n🔍 5个模型共有的reaction_id数量：{len(common_reactions)}")
#
#     for model in unified_data:
#         df = unified_data[model]
#         df = df[df['reaction_id'].isin(common_reactions)]
#         reaction_counts = df['reaction_id'].value_counts()
#         valid_reactions = reaction_counts[reaction_counts >= 20].index
#         df = df[df['reaction_id'].isin(valid_reactions)]
#         unified_data[model] = df.reset_index(drop=True)
#         print(f"✅ {model} 过滤完成：有效reaction数={len(valid_reactions)}，有效行数={len(df)}")
#     return unified_data
#
# def calculate_km_metrics(unified_data):
#     metrics_dict = {}
#     for model, df in unified_data.items():
#         model_metrics = {}
#         print(f"\n🔢 计算 {model} 的KM指标...")
#         for reaction_id, group in df.groupby('reaction_id'):
#             true_vals = group['true_log10_km'].values
#             pred_vals = group['model_pred_log10'].values
#             if len(np.unique(true_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
#                 continue
#             scc, _ = stats.spearmanr(true_vals, pred_vals)
#             scc = np.nan_to_num(scc)
#             pcc, _ = stats.pearsonr(true_vals, pred_vals)
#             pcc = np.nan_to_num(pcc)
#             indices = list(range(len(group)))
#             total_pairs = len(list(combinations(indices, 2)))
#             correct_pairs = 0
#             for i, j in combinations(indices, 2):
#                 true_i_higher = true_vals[i] > true_vals[j]
#                 pred_i_higher = pred_vals[i] > pred_vals[j]
#                 if true_i_higher == pred_i_higher:
#                     correct_pairs += 1
#             accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
#             model_metrics[reaction_id] = {'scc': scc, 'pcc': pcc, 'accuracy': accuracy}
#         metrics_dict[model] = model_metrics
#         print(f"✅ {model} 计算完成：有效reaction数={len(model_metrics)}")
#     return metrics_dict
#
# def organize_km_metrics(metrics_dict):
#     scc_data = {}
#     pcc_data = {}
#     acc_data = {}
#     models = ["my", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     for model in models:
#         model_metrics = metrics_dict.get(model, {})
#         scc_data[model] = [v['scc'] for v in model_metrics.values()] if model_metrics else []
#         pcc_data[model] = [v['pcc'] for v in model_metrics.values()] if model_metrics else []
#         acc_data[model] = [v['accuracy'] for v in model_metrics.values()] if model_metrics else []
#         print(f"📊 {model} | SCC={len(scc_data[model])}, PCC={len(pcc_data[model])}, ACC={len(acc_data[model])}")
#     return scc_data, pcc_data, acc_data
#
# # ======================================================================================
# #
# #   【3】kcat/Km 部分代码（完全不变）
# #
# # ======================================================================================
# kcat_km_paths = {
#     "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_kcat_km/kcat-km_N20_act_pred.pkl",
#     "my": "/mnt/usb1/wmx/catapro/analyse/kcat_km/kcat-km_data_filtered_N20_with_pred.pkl",
#     "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/kcat_km/kcat_km_N20_1.pkl",
#     "UniKP": "/mnt/usb1/wmx/unikp/analysis/kcat_km/kcat-km_N20.pkl",
#     "Eitlem": "/mnt/usb1/wmx/eitlem/kcat/N20_KKM_final_pred.pkl"
# }
#
# def load_data(file_paths):
#     data_dict = {}
#     for model, path in file_paths.items():
#         print(f"\n=== 加载 {model} 数据 ===")
#         df = pd.read_pickle(path)
#         df = df[(df['Km(M)'] > 0) & (df['reaction_id'].notnull())].reset_index(drop=True)
#         print(f"  有效行数：{len(df)}")
#         data_dict[model] = df
#     return data_dict
#
# def unify_scale(data_dict):
#     unified = {}
#     df = data_dict["CataPro"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['pred_log10[kcat/Km(s^-1mM^-1)]']
#     unified["CataPro"] = df
#
#     df = data_dict["my"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['pred_fusion_log10[kcat/Km]']
#     unified["my"] = df
#
#     df = data_dict["DLKcat"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['Pred_kcat_km_log2'] / np.log2(10)
#     unified["DLKcat"] = df
#
#     df = data_dict["UniKP"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['pred_log10_kcat_over_Km']
#     unified["UniKP"] = df
#
#     df_eit = data_dict["Eitlem"].copy()
#     df_eit['true_log10'] = np.log10(df_eit['kcat(s^-1)'] / df_eit['Km(M)'])
#     df_eit['pred_log10'] = df_eit['KKM_pred']
#     unified["Eitlem"] = df_eit
#
#     common = set(unified["my"]['reaction_id'])
#     for m in ["CataPro", "DLKcat", "UniKP", "Eitlem"]:
#         common &= set(unified[m]['reaction_id'])
#     print(f"\n共同 reaction 数量：{len(common)}")
#
#     for m in unified:
#         d = unified[m]
#         d = d[d['reaction_id'].isin(common)]
#         cnt = d['reaction_id'].value_counts()
#         valid = cnt[cnt >= 20].index
#         d = d[d['reaction_id'].isin(valid)]
#         unified[m] = d.reset_index(drop=True)
#         print(f"✅ {m} | 有效组：{len(valid)}，总行数：{len(d)}")
#     return unified
#
# def calc_metrics(unified):
#     metrics = {}
#     for model, df in unified.items():
#         model_res = {}
#         print(f"\n🔢 计算 {model} ...")
#         for rid, g in df.groupby("reaction_id"):
#             t = g['true_log10'].values
#             p = g['pred_log10'].values
#             if len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
#                 continue
#             scc, _ = stats.spearmanr(t, p)
#             pcc, _ = stats.pearsonr(t, p)
#             scc = np.nan_to_num(scc)
#             pcc = np.nan_to_num(pcc)
#             correct = 0
#             total = 0
#             for i, j in combinations(range(len(g)), 2):
#                 if (t[i] > t[j]) == (p[i] > p[j]):
#                     correct += 1
#                 total += 1
#             acc = correct / total if total > 0 else 0
#             model_res[rid] = {"scc": scc, "pcc": pcc, "accuracy": acc}
#         metrics[model] = model_res
#         print(f"✅ {model} 完成，有效数量：{len(model_res)}")
#     return metrics
#
# def arrange(metrics):
#     models = ["my", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     scc, pcc, acc = {}, {}, {}
#     for m in models:
#         data = list(metrics[m].values())
#         scc[m] = [x['scc'] for x in data]
#         pcc[m] = [x['pcc'] for x in data]
#         acc[m] = [x['accuracy'] for x in data]
#         print(f"📊 {m} | SCC={len(scc[m])}, PCC={len(pcc[m])}, ACC={len(acc[m])}")
#     return scc, pcc, acc
#
# # ======================================================================================
# #
# #   🔥🔥🔥 【核心：标题彻底修复 + 样式完美对齐】 🔥🔥🔥
# #
# # ======================================================================================
# def plot_all_in_one_3x3(kcat_scc, kcat_pcc, kcat_acc,
#                          km_scc, km_pcc, km_acc,
#                          kcatkm_scc, kcatkm_pcc, kcatkm_acc):
#     # 1. 字体建议
#     plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # 2. 关键：导出 SVG 时不将文本转为路径，防止重叠和无法选中
#     plt.rcParams['svg.fonttype'] = 'path'
#
#     # 3. 关键：统一数学字体集，stix 对 SVG 的兼容性最好，能防止下标错位
#     plt.rcParams['mathtext.fontset'] = 'stix'
#     # plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
#     # plt.rcParams['axes.unicode_minus'] = False
#     # plt.rcParams['svg.fonttype'] = 'none'
#     plt.rcParams['figure.dpi'] = 300
#     plt.rcParams['axes.titlepad'] = 10
#
#     colors = {"my": '#d62728', "CataPro": '#1f77b4', "DLKcat": '#ff7f0e', "UniKP": '#2ca02c', "Eitlem": "#9467bd"}
#     models = ["my", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     jitter_strength = 0.12
#
#     # 3行3列大图
#     fig, axes = plt.subplots(3, 3, figsize=(21, 18))
#
#     # 将整个字符串放入 $ $ 内部
#     # 使用 \mathrm{ } 或 \text{ } 包裹单词，确保字体和颜色在同一个渲染引擎下处理
#     plot_info = [
#         # 第1行：kcat
#         {"row": 0, "scc": kcat_scc, "pcc": kcat_pcc, "acc": kcat_acc,
#          "title": r"$\mathrm{k_{cat} \ dataset}$", "labels": ["a", "b", "c"]},
#         # 第2行：Km
#         {"row": 1, "scc": km_scc, "pcc": km_pcc, "acc": km_acc,
#          "title": r"$\mathrm{K_{m} \ dataset}$", "labels": ["d", "e", "f"]},
#         # 第3行：kcat/Km
#         {"row": 2, "scc": kcatkm_scc, "pcc": kcatkm_pcc, "acc": kcatkm_acc,
#          "title": r"$\mathrm{k_{cat}/K_{m} \ dataset}$", "labels": ["g", "h", "i"]},
#     ]
#
#     # 统一绘图函数
#     def draw_subplot(ax, data, title, ylabel, ylim, label, zero=False):
#         valid_data, valid_labels, curr_colors = [], [], []
#         for m in models:
#             if len(data[m]) > 0:
#                 valid_data.append(data[m])
#                 valid_labels.append(m)
#                 curr_colors.append(colors[m])
#
#         # 绘制箱型图（匹配参考图样式）
#         box = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, showmeans=True,
#                          widths=0.6, zorder=2,
#                          medianprops={'color': 'white', 'linewidth': 2},
#                          meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5},
#                          flierprops={'marker': '.'})
#         for patch, c in zip(box['boxes'], curr_colors):
#             patch.set_facecolor(c)
#             patch.set_alpha(0.8)
#
#         # 叠加黑色抖动散点
#         for i, m in enumerate(valid_labels):
#             y = data[m]
#             x = np.random.normal(i + 1, jitter_strength, len(y))
#             ax.scatter(x, y, c='black', alpha=0.4, s=8, zorder=3, linewidths=0)
#
#         # 细节优化
#         ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
#         ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
#         ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
#         ax.set_ylim(ylim)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         if zero:
#             ax.axhline(0, color='black', lw=1, alpha=0.2, zorder=1)
#
#         # 子图标注（紧贴左上角）
#         ax.text(-0.08, 1.03, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
#
#     # 批量绘制
#     for info in plot_info:
#         row = info["row"]
#         draw_subplot(axes[row, 0], info["scc"], info["title"], "Spearman Correlation", (-1.05, 1.05), info["labels"][0], zero=True)
#         draw_subplot(axes[row, 1], info["pcc"], info["title"], "Pearson Correlation", (-1.05, 1.05), info["labels"][1], zero=True)
#         draw_subplot(axes[row, 2], info["acc"], info["title"], "Pairwise Accuracy", (0, 1.05), info["labels"][2])
#
#     plt.tight_layout(pad=3.0)
#     # 保存SVG + PNG
#     plt.savefig(r"/mnt/usb1/wmx/catapro/analyse/all_3x3_figure_title_fixed.svg", bbox_inches='tight', dpi=300)
#     #plt.savefig(r"/mnt/usb1/wmx/catapro/analyse/all_3x3_figure_title_fixed.png", bbox_inches='tight', dpi=300)
#     plt.show()
#     print("✅ 标题错乱问题彻底解决！3×3大图已保存")
#
# # ======================================================================================
# #
# #   主执行流程
# #
# # ======================================================================================
# if __name__ == "__main__":
#     np.seterr(invalid='ignore', divide='ignore')
#
#     # 1. kcat
#     data_20 = load_kcat_data(file_config["N≥20"], 20)
#     unified_20 = unify_log10_scale(data_20, 20)
#     metrics_20 = calculate_metrics_per_reaction(unified_20)
#     scc_20, pcc_20, acc_20 = organize_metrics(metrics_20)
#
#     # 2. Km
#     data_km = load_and_clean_km_data(km_file_paths)
#     unified_km = unify_km_log10_scale(data_km)
#     metrics_km = calculate_km_metrics(unified_km)
#     scc_km, pcc_km, acc_km = organize_km_metrics(metrics_km)
#
#     # 3. kcat/Km
#     data_kcatkm = load_data(kcat_km_paths)
#     unified_kcatkm = unify_scale(data_kcatkm)
#     metrics_kcatkm = calc_metrics(unified_kcatkm)
#     scc_kcatkm, pcc_kcatkm, acc_kcatkm = arrange(metrics_kcatkm)
#
#     # 4. 绘制修复后的大图
#     plot_all_in_one_3x3(
#         scc_20, pcc_20, acc_20,
#         scc_km, pcc_km, acc_km,
#         scc_kcatkm, pcc_kcatkm, acc_kcatkm
#     )
#
#     print("\n🎉 全部完成！标题再也不会错乱了")
#
# # ===================== 新增：全指标统计汇总（论文直接用）=====================
# import pandas as pd
#
# def get_stat_summary(data_dict, name):
#     """
#     输入模型指标字典，输出每个模型的完整统计量：
#     count, mean, median, q1(25%), q3(75%), std, min, max
#     """
#     models = ["my", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     res = []
#     for m in models:
#         arr = np.array(data_dict[m])
#         if len(arr) == 0:
#             res.append({
#                 "Dataset_Metric": name,
#                 "Model": m,
#                 "n": 0,
#                 "mean": np.nan,
#                 "median": np.nan,
#                 "Q1": np.nan,
#                 "Q3": np.nan,
#                 "std": np.nan,
#                 "min": np.nan,
#                 "max": np.nan
#             })
#             continue
#         res.append({
#             "Dataset_Metric": name,
#             "Model": m,
#             "n": len(arr),
#             "mean": round(np.mean(arr), 4),
#             "median": round(np.median(arr), 4),
#             "Q1": round(np.percentile(arr, 25), 4),
#             "Q3": round(np.percentile(arr, 75), 4),
#             "std": round(np.std(arr, ddof=1), 4),
#             "min": round(np.min(arr), 4),
#             "max": round(np.max(arr), 4)
#         })
#     return pd.DataFrame(res)
#
# # ========== 批量统计所有 9 组指标 ==========
# # 1. kcat
# df_kcat_scc = get_stat_summary(scc_20, "kcat_SCC")
# df_kcat_pcc = get_stat_summary(pcc_20, "kcat_PCC")
# df_kcat_acc = get_stat_summary(acc_20, "kcat_Accuracy")
#
# # 2. Km
# df_km_scc = get_stat_summary(scc_km, "Km_SCC")
# df_km_pcc = get_stat_summary(scc_km, "Km_PCC")
# df_km_acc = get_stat_summary(scc_km, "Km_Accuracy")
#
# # 3. kcat/Km
# df_kkm_scc = get_stat_summary(scc_kcatkm, "kcat/Km_SCC")
# df_kkm_pcc = get_stat_summary(scc_kcatkm, "kcat/Km_PCC")
# df_kkm_acc = get_stat_summary(scc_kcatkm, "kcat/Km_Accuracy")
#
# # 全部合并总表
# all_stats = pd.concat([
#     df_kcat_scc, df_kcat_pcc, df_kcat_acc,
#     df_km_scc, df_km_pcc, df_km_acc,
#     df_kkm_scc, df_kkm_pcc, df_kkm_acc
# ], ignore_index=True)
#
# # 控制台完整打印
# print("\n" + "="*120)
# print("【完整统计汇总表：均值、中位数、四分位数Q1/Q3、标准差、极值、样本量n】")
# print("="*120)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# print(all_stats)
#
# # 保存为csv，方便论文制表
# all_stats.to_csv("/mnt/usb1/wmx/catapro/analyse/model_metrics_full_statistics.csv", index=False)
# print("\n✅ 完整统计结果已保存CSV：/mnt/usb1/wmx/catapro/analyse/model_metrics_full_statistics.csv")


# import pandas as pd
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from itertools import combinations
# from scikit_posthocs import posthoc_dunn
# import os
# import random
#
# # ======================================================================================
# #
# #   【1】kcat 部分代码（my → DB-Kinetic）
# #
# # ======================================================================================
# file_config = {
#     "N≥20": {
#         "DB-Kinetic": "/mnt/usb1/wmx/catapro/analyse/kcat/kcat_pred_by_fold_20.pkl",
#         "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_kcat/N20.pkl",
#         "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/kcat/N20_1.pkl",
#         "UniKP": "/mnt/usb1/wmx/unikp/analysis/kcat/N20.pkl",
#         "Eitlem": "/mnt/usb1/wmx/eitlem/kcat/kcat_N20_final_pred.pkl"
#     }
# }
#
# def load_kcat_data(file_paths, n_threshold):
#     data_dict = {}
#     for model, path in file_paths.items():
#         print(f"\n=== 加载 {model} 数据（{n_threshold}）===")
#         if not os.path.exists(path):
#             print(f"❌ 错误：{model} 文件不存在 → {path}")
#             raise FileNotFoundError(f"Missing file for {model}: {path}")
#         try:
#             df = pd.read_pickle(path)
#             df = df[df['kcat(s^-1)'] > 0].reset_index(drop=True)
#             data_dict[model] = df
#             print(f"✅ {model} 数据加载完成，有效行数：{len(df)}")
#         except Exception as e:
#             print(f"❌ 读取 {model} 失败：{e}")
#             raise e
#     return data_dict
#
# def unify_log10_scale(data_dict, n_threshold):
#     unified_data = {}
#     df_my = data_dict["DB-Kinetic"].copy()
#     df_my['true_log10_kcat'] = np.log10(df_my['kcat(s^-1)'])
#     df_my.rename(columns={'pred_log10_kcat': 'model_pred_log10'}, inplace=True)
#     unified_data["DB-Kinetic"] = df_my
#
#     df_cata = data_dict["CataPro"].copy()
#     df_cata['true_log10_kcat'] = np.log10(df_cata['kcat(s^-1)'])
#     df_cata.rename(columns={'pred_log10[kcat(s^-1)]': 'model_pred_log10'}, inplace=True)
#     unified_data["CataPro"] = df_cata
#
#     df_dlk = data_dict["DLKcat"].copy()
#     df_dlk['true_log10_kcat'] = np.log10(df_dlk['kcat(s^-1)'])
#     if 'Pred_Kcat_log2' in df_dlk.columns:
#         df_dlk['model_pred_log10'] = df_dlk['Pred_Kcat_log2'] / np.log2(10)
#     else:
#         df_dlk['model_pred_log10'] = df_dlk.get('pred_log10_kcat', np.nan)
#     unified_data["DLKcat"] = df_dlk
#
#     df_uni = data_dict["UniKP"].copy()
#     df_uni['true_log10_kcat'] = np.log10(df_uni['kcat(s^-1)'])
#     df_uni.rename(columns={'pred_log10_kcat': 'model_pred_log10'}, inplace=True)
#     unified_data["UniKP"] = df_uni
#
#     df_eit = data_dict["Eitlem"].copy()
#     df_eit['true_log10_kcat'] = np.log10(df_eit['kcat(s^-1)'])
#     df_eit['model_pred_log10'] = df_eit['pred']
#     unified_data["Eitlem"] = df_eit
#
#     common_reactions = set(unified_data["DB-Kinetic"]['reaction_id'])
#     for model in ["CataPro", "DLKcat", "UniKP", "Eitlem"]:
#         if model in unified_data and not unified_data[model].empty:
#             common_reactions = common_reactions & set(unified_data[model]['reaction_id'])
#     print(f"\n🔍 5个模型共有的reaction_id数量：{len(common_reactions)}")
#
#     for model in unified_data:
#         df = unified_data[model]
#         if df.empty:
#             unified_data[model] = df
#             print(f"⚠️ {model} 原始数据为空，跳过过滤")
#             continue
#         df = df[df['reaction_id'].isin(common_reactions)]
#         reaction_counts = df['reaction_id'].value_counts()
#         valid_reactions = reaction_counts[reaction_counts >= n_threshold].index
#         df = df[df['reaction_id'].isin(valid_reactions)]
#         unified_data[model] = df.reset_index(drop=True)
#         print(f"✅ {model} 数据过滤完成，有效reaction数：{len(valid_reactions)}，有效行数：{len(df)}")
#     return unified_data
#
# def calculate_metrics_per_reaction(unified_data):
#     metrics_dict = {}
#     for model, df in unified_data.items():
#         model_metrics = {}
#         print(f"\n🔢 计算 {model} 模型指标...")
#         if df.empty:
#             print(f"⚠️ {model} 数据为空，跳过指标计算")
#             metrics_dict[model] = model_metrics
#             continue
#         for idx, (reaction_id, group) in enumerate(df.groupby('reaction_id')):
#             true_vals = group['true_log10_kcat'].values
#             pred_vals = group['model_pred_log10'].values
#             if len(np.unique(true_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
#                 continue
#             scc, _ = stats.spearmanr(true_vals, pred_vals)
#             scc = np.nan_to_num(scc)
#             pcc, _ = stats.pearsonr(true_vals, pred_vals)
#             pcc = np.nan_to_num(pcc)
#             indices = list(range(len(group)))
#             combinations_list = list(combinations(indices, 2))
#             correct = 0
#             for i, j in combinations_list:
#                 true_i_higher = true_vals[i] > true_vals[j]
#                 pred_i_higher = pred_vals[i] > pred_vals[j]
#                 if true_i_higher == pred_i_higher:
#                     correct += 1
#             accuracy = correct / len(combinations_list) if len(combinations_list) > 0 else 0
#             model_metrics[reaction_id] = {'scc': scc, 'pcc': pcc, 'accuracy': accuracy}
#         metrics_dict[model] = model_metrics
#         print(f"✅ {model} 指标计算完成，有效reaction数：{len(model_metrics)}")
#     return metrics_dict
#
# def organize_metrics(metrics_dict):
#     scc_data = {}
#     pcc_data = {}
#     acc_data = {}
#     models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     for model in models:
#         model_metrics = metrics_dict.get(model, {})
#         scc_data[model] = [v['scc'] for v in model_metrics.values()] if model_metrics else []
#         pcc_data[model] = [v['pcc'] for v in model_metrics.values()] if model_metrics else []
#         acc_data[model] = [v['accuracy'] for v in model_metrics.values()] if model_metrics else []
#         print(f"📊 {model} - SCC：{len(scc_data[model])}, PCC：{len(pcc_data[model])}, 准确率：{len(acc_data[model])}")
#     return scc_data, pcc_data, acc_data
#
# # ======================================================================================
# #
# #   【2】Km 部分代码（my → DB-Kinetic）
# #
# # ======================================================================================
# km_file_paths = {
#     "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_km/km20.pkl",
#     "DB-Kinetic": "/mnt/usb1/wmx/catapro/analyse/km/km-data_filtered_N20_with_pred.pkl",
#     "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/km/km_N20_matched_result.pkl",
#     "UniKP": "/mnt/usb1/wmx/unikp/analysis/km/km_N20.pkl",
#     "Eitlem": "/mnt/usb1/wmx/eitlem/kcat/km_N20_final_pred.pkl"
# }
#
# def load_and_clean_km_data(file_paths):
#     data_dict = {}
#     for model, path in file_paths.items():
#         print(f"\n=== 加载 {model} 数据 ===")
#         df = pd.read_pickle(path)
#         df = df[(df['Km(M)'] > 0) & (df['reaction_id'].notnull())].reset_index(drop=True)
#         print(f"   清洗后总行数：{len(df)}")
#         data_dict[model] = df
#     return data_dict
#
# def unify_km_log10_scale(data_dict):
#     unified_data = {}
#     df_cata = data_dict["CataPro"].copy()
#     df_cata['true_log10_km'] = df_cata['log10_Km']
#     df_cata['model_pred_log10'] = df_cata['pred_log10[Km(mM)]']
#     unified_data["CataPro"] = df_cata
#
#     df_my = data_dict["DB-Kinetic"].copy()
#     df_my['true_log10_km'] = df_my['log10_Km']
#     df_my['model_pred_log10'] = df_my['pred_log10[Km(mM)]']
#     unified_data["DB-Kinetic"] = df_my
#
#     df_dlk = data_dict["DLKcat"].copy()
#     df_dlk['true_log10_km'] = df_dlk['log10_Km']
#     df_dlk['model_pred_log10'] = df_dlk['Pred_Km_log2'] / np.log2(10)
#     unified_data["DLKcat"] = df_dlk
#
#     df_uni = data_dict["UniKP"].copy()
#     df_uni['true_log10_km'] = df_uni['log10_Km']
#     df_uni['model_pred_log10'] = df_uni['pred_log10_km']
#     unified_data["UniKP"] = df_uni
#
#     df_eit = data_dict["Eitlem"].copy()
#     df_eit['true_log10_km'] = np.log10(df_eit['Km(M)'])
#     df_eit['model_pred_log10'] = df_eit['pred']
#     unified_data["Eitlem"] = df_eit
#
#     common_reactions = set(unified_data["DB-Kinetic"]['reaction_id'])
#     for model in ["CataPro", "DLKcat", "UniKP", "Eitlem"]:
#         common_reactions = common_reactions & set(unified_data[model]['reaction_id'])
#     print(f"\n🔍 5个模型共有的reaction_id数量：{len(common_reactions)}")
#
#     for model in unified_data:
#         df = unified_data[model]
#         df = df[df['reaction_id'].isin(common_reactions)]
#         reaction_counts = df['reaction_id'].value_counts()
#         valid_reactions = reaction_counts[reaction_counts >= 20].index
#         df = df[df['reaction_id'].isin(valid_reactions)]
#         unified_data[model] = df.reset_index(drop=True)
#         print(f"✅ {model} 过滤完成：有效reaction数={len(valid_reactions)}，有效行数={len(df)}")
#     return unified_data
#
# def calculate_km_metrics(unified_data):
#     metrics_dict = {}
#     for model, df in unified_data.items():
#         model_metrics = {}
#         print(f"\n🔢 计算 {model} 的KM指标...")
#         for reaction_id, group in df.groupby('reaction_id'):
#             true_vals = group['true_log10_km'].values
#             pred_vals = group['model_pred_log10'].values
#             if len(np.unique(true_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
#                 continue
#             scc, _ = stats.spearmanr(true_vals, pred_vals)
#             scc = np.nan_to_num(scc)
#             pcc, _ = stats.pearsonr(true_vals, pred_vals)
#             pcc = np.nan_to_num(pcc)
#             indices = list(range(len(group)))
#             total_pairs = len(list(combinations(indices, 2)))
#             correct_pairs = 0
#             for i, j in combinations(indices, 2):
#                 true_i_higher = true_vals[i] > true_vals[j]
#                 pred_i_higher = pred_vals[i] > pred_vals[j]
#                 if true_i_higher == pred_i_higher:
#                     correct_pairs += 1
#             accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
#             model_metrics[reaction_id] = {'scc': scc, 'pcc': pcc, 'accuracy': accuracy}
#         metrics_dict[model] = model_metrics
#         print(f"✅ {model} 计算完成：有效reaction数={len(model_metrics)}")
#     return metrics_dict
#
# def organize_km_metrics(metrics_dict):
#     scc_data = {}
#     pcc_data = {}
#     acc_data = {}
#     models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     for model in models:
#         model_metrics = metrics_dict.get(model, {})
#         scc_data[model] = [v['scc'] for v in model_metrics.values()] if model_metrics else []
#         pcc_data[model] = [v['pcc'] for v in model_metrics.values()] if model_metrics else []
#         acc_data[model] = [v['accuracy'] for v in model_metrics.values()] if model_metrics else []
#         print(f"📊 {model} | SCC={len(scc_data[model])}, PCC={len(pcc_data[model])}, ACC={len(acc_data[model])}")
#     return scc_data, pcc_data, acc_data
#
# # ======================================================================================
# #
# #   【3】kcat/Km 部分代码（my → DB-Kinetic）
# #
# # ======================================================================================
# kcat_km_paths = {
#     "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_kcat_km/kcat-km_N20_act_pred.pkl",
#     "DB-Kinetic": "/mnt/usb1/wmx/catapro/analyse/kcat_km/kcat-km_data_filtered_N20_with_pred.pkl",
#     "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/kcat_km/kcat_km_N20_1.pkl",
#     "UniKP": "/mnt/usb1/wmx/unikp/analysis/kcat_km/kcat-km_N20.pkl",
#     "Eitlem": "/mnt/usb1/wmx/eitlem/kcat/N20_KKM_final_pred.pkl"
# }
#
# def load_data(file_paths):
#     data_dict = {}
#     for model, path in file_paths.items():
#         print(f"\n=== 加载 {model} 数据 ===")
#         df = pd.read_pickle(path)
#         df = df[(df['Km(M)'] > 0) & (df['reaction_id'].notnull())].reset_index(drop=True)
#         print(f"  有效行数：{len(df)}")
#         data_dict[model] = df
#     return data_dict
#
# def unify_scale(data_dict):
#     unified = {}
#     df = data_dict["CataPro"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['pred_log10[kcat/Km(s^-1mM^-1)]']
#     unified["CataPro"] = df
#
#     df = data_dict["DB-Kinetic"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['pred_fusion_log10[kcat/Km]']
#     unified["DB-Kinetic"] = df
#
#     df = data_dict["DLKcat"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['Pred_kcat_km_log2'] / np.log2(10)
#     unified["DLKcat"] = df
#
#     df = data_dict["UniKP"].copy()
#     df['true_log10'] = df['log10_kcat_over_Km']
#     df['pred_log10'] = df['pred_log10_kcat_over_Km']
#     unified["UniKP"] = df
#
#     df_eit = data_dict["Eitlem"].copy()
#     df_eit['true_log10'] = np.log10(df_eit['kcat(s^-1)'] / df_eit['Km(M)'])
#     df_eit['pred_log10'] = df_eit['KKM_pred']
#     unified["Eitlem"] = df_eit
#
#     common = set(unified["DB-Kinetic"]['reaction_id'])
#     for m in ["CataPro", "DLKcat", "UniKP", "Eitlem"]:
#         common &= set(unified[m]['reaction_id'])
#     print(f"\n共同 reaction 数量：{len(common)}")
#
#     for m in unified:
#         d = unified[m]
#         d = d[d['reaction_id'].isin(common)]
#         cnt = d['reaction_id'].value_counts()
#         valid = cnt[cnt >= 20].index
#         d = d[d['reaction_id'].isin(valid)]
#         unified[m] = d.reset_index(drop=True)
#         print(f"✅ {m} | 有效组：{len(valid)}，总行数：{len(d)}")
#     return unified
#
# def calc_metrics(unified):
#     metrics = {}
#     for model, df in unified.items():
#         model_res = {}
#         print(f"\n🔢 计算 {model} ...")
#         for rid, g in df.groupby("reaction_id"):
#             t = g['true_log10'].values
#             p = g['pred_log10'].values
#             if len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
#                 continue
#             scc, _ = stats.spearmanr(t, p)
#             pcc, _ = stats.pearsonr(t, p)
#             scc = np.nan_to_num(scc)
#             pcc = np.nan_to_num(pcc)
#             correct = 0
#             total = 0
#             for i, j in combinations(range(len(g)), 2):
#                 if (t[i] > t[j]) == (p[i] > p[j]):
#                     correct += 1
#                 total += 1
#             acc = correct / total if total > 0 else 0
#             model_res[rid] = {"scc": scc, "pcc": pcc, "accuracy": acc}
#         metrics[model] = model_res
#         print(f"✅ {model} 完成，有效数量：{len(model_res)}")
#     return metrics
#
# def arrange(metrics):
#     models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     scc, pcc, acc = {}, {}, {}
#     for m in models:
#         data = list(metrics[m].values())
#         scc[m] = [x['scc'] for x in data]
#         pcc[m] = [x['pcc'] for x in data]
#         acc[m] = [x['accuracy'] for x in data]
#         print(f"📊 {m} | SCC={len(scc[m])}, PCC={len(pcc[m])}, ACC={len(acc[m])}")
#     return scc, pcc, acc
#
# # ======================================================================================
# #
# #   绘图部分（已放大横轴模型字体）
# #
# # ======================================================================================
# def plot_all_in_one_3x3(kcat_scc, kcat_pcc, kcat_acc,
#                          km_scc, km_pcc, km_acc,
#                          kcatkm_scc, kcatkm_pcc, kcatkm_acc):
#     plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['svg.fonttype'] = 'path'
#     plt.rcParams['mathtext.fontset'] = 'stix'
#     plt.rcParams['figure.dpi'] = 300
#     plt.rcParams['axes.titlepad'] = 10
#
#     colors = {"DB-Kinetic": '#d62728', "CataPro": '#1f77b4', "DLKcat": '#ff7f0e', "UniKP": '#2ca02c', "Eitlem": "#9467bd"}
#     models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     jitter_strength = 0.12
#
#     fig, axes = plt.subplots(3, 3, figsize=(21, 18))
#
#     plot_info = [
#         {"row": 0, "scc": kcat_scc, "pcc": kcat_pcc, "acc": kcat_acc,
#          "title": r"$\mathrm{k_{cat} \ dataset}$", "labels": ["a", "b", "c"]},
#         {"row": 1, "scc": km_scc, "pcc": km_pcc, "acc": km_acc,
#          "title": r"$\mathrm{K_{m} \ dataset}$", "labels": ["d", "e", "f"]},
#         {"row": 2, "scc": kcatkm_scc, "pcc": kcatkm_pcc, "acc": kcatkm_acc,
#          "title": r"$\mathrm{k_{cat}/K_{m} \ dataset}$", "labels": ["g", "h", "i"]},
#     ]
#
#     def draw_subplot(ax, data, title, ylabel, ylim, label, zero=False):
#         valid_data, valid_labels, curr_colors = [], [], []
#         for m in models:
#             if len(data[m]) > 0:
#                 valid_data.append(data[m])
#                 valid_labels.append(m)
#                 curr_colors.append(colors[m])
#
#         box = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, showmeans=True,
#                          widths=0.6, zorder=2,
#                          medianprops={'color': 'white', 'linewidth': 2},
#                          meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5},
#                          flierprops={'marker': '.'})
#         for patch, c in zip(box['boxes'], curr_colors):
#             patch.set_facecolor(c)
#             patch.set_alpha(0.8)
#
#         for i, m in enumerate(valid_labels):
#             y = data[m]
#             x = np.random.normal(i + 1, jitter_strength, len(y))
#             ax.scatter(x, y, c='black', alpha=0.4, s=8, zorder=3, linewidths=0)
#
#         ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
#         ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
#         ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
#         ax.set_ylim(ylim)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         if zero:
#             ax.axhline(0, color='black', lw=1, alpha=0.2, zorder=1)
#
#         # ---------- 放大横轴模型字体 ----------
#         ax.tick_params(axis='x', labelsize=13)  # 👈 就是这一行！
#
#         ax.text(-0.04, 1.01, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
#
#     for info in plot_info:
#         row = info["row"]
#         draw_subplot(axes[row, 0], info["scc"], info["title"], "Spearman Correlation", (-1.05, 1.05), info["labels"][0], zero=True)
#         draw_subplot(axes[row, 1], info["pcc"], info["title"], "Pearson Correlation", (-1.05, 1.05), info["labels"][1], zero=True)
#         draw_subplot(axes[row, 2], info["acc"], info["title"], "Pairwise Accuracy", (0, 1.05), info["labels"][2])
#
#     plt.tight_layout(pad=3.0)
#     plt.savefig(r"/mnt/usb1/wmx/catapro/analyse/all_3x3_figure_title_fixed.svg", bbox_inches='tight', dpi=300)
#     plt.show()
#     print("✅ 3×3大图已保存！横轴模型字体已放大")
#
# # ======================================================================================
# #
# #   主执行流程
# #
# # ======================================================================================
# if __name__ == "__main__":
#     np.seterr(invalid='ignore', divide='ignore')
#
#     # 1. kcat
#     data_20 = load_kcat_data(file_config["N≥20"], 20)
#     unified_20 = unify_log10_scale(data_20, 20)
#     metrics_20 = calculate_metrics_per_reaction(unified_20)
#     scc_20, pcc_20, acc_20 = organize_metrics(metrics_20)
#
#     # 2. Km
#     data_km = load_and_clean_km_data(km_file_paths)
#     unified_km = unify_km_log10_scale(data_km)
#     metrics_km = calculate_km_metrics(unified_km)
#     scc_km, pcc_km, acc_km = organize_km_metrics(metrics_km)
#
#     # 3. kcat/Km
#     data_kcatkm = load_data(kcat_km_paths)
#     unified_kcatkm = unify_scale(data_kcatkm)
#     metrics_kcatkm = calc_metrics(unified_kcatkm)
#     scc_kcatkm, pcc_kcatkm, acc_kcatkm = arrange(metrics_kcatkm)
#
#     # 绘图
#     plot_all_in_one_3x3(
#         scc_20, pcc_20, acc_20,
#         scc_km, pcc_km, acc_km,
#         scc_kcatkm, pcc_kcatkm, acc_kcatkm
#     )
#
# # ===================== 统计汇总 =====================
# def get_stat_summary(data_dict, name):
#     models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "Eitlem"]
#     res = []
#     for m in models:
#         arr = np.array(data_dict[m])
#         if len(arr) == 0:
#             res.append({
#                 "Dataset_Metric": name,
#                 "Model": m,
#                 "n": 0,
#                 "mean": np.nan,
#                 "median": np.nan,
#                 "Q1": np.nan,
#                 "Q3": np.nan,
#                 "std": np.nan,
#                 "min": np.nan,
#                 "max": np.nan
#             })
#             continue
#         res.append({
#             "Dataset_Metric": name,
#             "Model": m,
#             "n": len(arr),
#             "mean": round(np.mean(arr), 4),
#             "median": round(np.median(arr), 4),
#             "Q1": round(np.percentile(arr, 25), 4),
#             "Q3": round(np.percentile(arr, 75), 4),
#             "std": round(np.std(arr, ddof=1), 4),
#             "min": round(np.min(arr), 4),
#             "max": round(np.max(arr), 4)
#         })
#     return pd.DataFrame(res)

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from itertools import combinations
from scikit_posthocs import posthoc_dunn
import os
import random

# ======================================================================================
#
#   【1】kcat 部分代码（my → DB-Kinetic）
#
# ======================================================================================
file_config = {
    "N≥20": {
        "DB-Kinetic": "/mnt/usb1/wmx/catapro/analyse/kcat/kcat_pred_by_fold_20.pkl",
        "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_kcat/N20.pkl",
        "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/kcat/N20_1.pkl",
        "UniKP": "/mnt/usb1/wmx/unikp/analysis/kcat/N20.pkl",
        "EITLEM": "/mnt/usb1/wmx/eitlem/kcat/kcat_N20_final_pred.pkl"
    }
}

def load_kcat_data(file_paths, n_threshold):
    data_dict = {}
    for model, path in file_paths.items():
        print(f"\n=== 加载 {model} 数据（{n_threshold}）===")
        if not os.path.exists(path):
            print(f"❌ 错误：{model} 文件不存在 → {path}")
            raise FileNotFoundError(f"Missing file for {model}: {path}")
        try:
            df = pd.read_pickle(path)
            df = df[df['kcat(s^-1)'] > 0].reset_index(drop=True)
            data_dict[model] = df
            print(f"✅ {model} 数据加载完成，有效行数：{len(df)}")
        except Exception as e:
            print(f"❌ 读取 {model} 失败：{e}")
            raise e
    return data_dict

def unify_log10_scale(data_dict, n_threshold):
    unified_data = {}
    df_my = data_dict["DB-Kinetic"].copy()
    df_my['true_log10_kcat'] = np.log10(df_my['kcat(s^-1)'])
    df_my.rename(columns={'pred_log10_kcat': 'model_pred_log10'}, inplace=True)
    unified_data["DB-Kinetic"] = df_my

    df_cata = data_dict["CataPro"].copy()
    df_cata['true_log10_kcat'] = np.log10(df_cata['kcat(s^-1)'])
    df_cata.rename(columns={'pred_log10[kcat(s^-1)]': 'model_pred_log10'}, inplace=True)
    unified_data["CataPro"] = df_cata

    df_dlk = data_dict["DLKcat"].copy()
    df_dlk['true_log10_kcat'] = np.log10(df_dlk['kcat(s^-1)'])
    if 'Pred_Kcat_log2' in df_dlk.columns:
        df_dlk['model_pred_log10'] = df_dlk['Pred_Kcat_log2'] / np.log2(10)
    else:
        df_dlk['model_pred_log10'] = df_dlk.get('pred_log10_kcat', np.nan)
    unified_data["DLKcat"] = df_dlk

    df_uni = data_dict["UniKP"].copy()
    df_uni['true_log10_kcat'] = np.log10(df_uni['kcat(s^-1)'])
    df_uni.rename(columns={'pred_log10_kcat': 'model_pred_log10'}, inplace=True)
    unified_data["UniKP"] = df_uni

    df_eit = data_dict["EITLEM"].copy()
    df_eit['true_log10_kcat'] = np.log10(df_eit['kcat(s^-1)'])
    df_eit['model_pred_log10'] = df_eit['pred']
    unified_data["EITLEM"] = df_eit

    common_reactions = set(unified_data["DB-Kinetic"]['reaction_id'])
    for model in ["CataPro", "DLKcat", "UniKP", "EITLEM"]:
        if model in unified_data and not unified_data[model].empty:
            common_reactions = common_reactions & set(unified_data[model]['reaction_id'])
    print(f"\n🔍 5个模型共有的reaction_id数量：{len(common_reactions)}")

    for model in unified_data:
        df = unified_data[model]
        if df.empty:
            unified_data[model] = df
            print(f"⚠️ {model} 原始数据为空，跳过过滤")
            continue
        df = df[df['reaction_id'].isin(common_reactions)]
        reaction_counts = df['reaction_id'].value_counts()
        valid_reactions = reaction_counts[reaction_counts >= n_threshold].index
        df = df[df['reaction_id'].isin(valid_reactions)]
        unified_data[model] = df.reset_index(drop=True)
        print(f"✅ {model} 数据过滤完成，有效reaction数：{len(valid_reactions)}，有效行数：{len(df)}")
    return unified_data

def calculate_metrics_per_reaction(unified_data):
    metrics_dict = {}
    for model, df in unified_data.items():
        model_metrics = {}
        print(f"\n🔢 计算 {model} 模型指标...")
        if df.empty:
            print(f"⚠️ {model} 数据为空，跳过指标计算")
            metrics_dict[model] = model_metrics
            continue
        for idx, (reaction_id, group) in enumerate(df.groupby('reaction_id')):
            true_vals = group['true_log10_kcat'].values
            pred_vals = group['model_pred_log10'].values
            if len(np.unique(true_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
                continue
            scc, _ = stats.spearmanr(true_vals, pred_vals)
            scc = np.nan_to_num(scc)
            pcc, _ = stats.pearsonr(true_vals, pred_vals)
            pcc = np.nan_to_num(pcc)
            indices = list(range(len(group)))
            combinations_list = list(combinations(indices, 2))
            correct = 0
            for i, j in combinations_list:
                true_i_higher = true_vals[i] > true_vals[j]
                pred_i_higher = pred_vals[i] > pred_vals[j]
                if true_i_higher == pred_i_higher:
                    correct += 1
            accuracy = correct / len(combinations_list) if len(combinations_list) > 0 else 0
            model_metrics[reaction_id] = {'scc': scc, 'pcc': pcc, 'accuracy': accuracy}
        metrics_dict[model] = model_metrics
        print(f"✅ {model} 指标计算完成，有效reaction数：{len(model_metrics)}")
    return metrics_dict

def organize_metrics(metrics_dict):
    scc_data = {}
    pcc_data = {}
    acc_data = {}
    models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "EITLEM"]
    for model in models:
        model_metrics = metrics_dict.get(model, {})
        scc_data[model] = [v['scc'] for v in model_metrics.values()] if model_metrics else []
        pcc_data[model] = [v['pcc'] for v in model_metrics.values()] if model_metrics else []
        acc_data[model] = [v['accuracy'] for v in model_metrics.values()] if model_metrics else []
        print(f"📊 {model} - SCC：{len(scc_data[model])}, PCC：{len(pcc_data[model])}, 准确率：{len(acc_data[model])}")
    return scc_data, pcc_data, acc_data

# ======================================================================================
#
#   【2】Km 部分代码（my → DB-Kinetic）
#
# ======================================================================================
km_file_paths = {
    "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_km/km20.pkl",
    "DB-Kinetic": "/mnt/usb1/wmx/catapro/analyse/km/km-data_filtered_N20_with_pred.pkl",
    "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/km/km_N20_matched_result.pkl",
    "UniKP": "/mnt/usb1/wmx/unikp/analysis/km/km_N20.pkl",
    "EITLEM": "/mnt/usb1/wmx/eitlem/kcat/km_N20_final_pred.pkl"
}

def load_and_clean_km_data(file_paths):
    data_dict = {}
    for model, path in file_paths.items():
        print(f"\n=== 加载 {model} 数据 ===")
        df = pd.read_pickle(path)
        df = df[(df['Km(M)'] > 0) & (df['reaction_id'].notnull())].reset_index(drop=True)
        print(f"   清洗后总行数：{len(df)}")
        data_dict[model] = df
    return data_dict

def unify_km_log10_scale(data_dict):
    unified_data = {}
    df_cata = data_dict["CataPro"].copy()
    df_cata['true_log10_km'] = df_cata['log10_Km']
    df_cata['model_pred_log10'] = df_cata['pred_log10[Km(mM)]']
    unified_data["CataPro"] = df_cata

    df_my = data_dict["DB-Kinetic"].copy()
    df_my['true_log10_km'] = df_my['log10_Km']
    df_my['model_pred_log10'] = df_my['pred_log10[Km(mM)]']
    unified_data["DB-Kinetic"] = df_my

    df_dlk = data_dict["DLKcat"].copy()
    df_dlk['true_log10_km'] = df_dlk['log10_Km']
    df_dlk['model_pred_log10'] = df_dlk['Pred_Km_log2'] / np.log2(10)
    unified_data["DLKcat"] = df_dlk

    df_uni = data_dict["UniKP"].copy()
    df_uni['true_log10_km'] = df_uni['log10_Km']
    df_uni['model_pred_log10'] = df_uni['pred_log10_km']
    unified_data["UniKP"] = df_uni

    df_eit = data_dict["EITLEM"].copy()
    df_eit['true_log10_km'] = np.log10(df_eit['Km(M)'])
    df_eit['model_pred_log10'] = df_eit['pred']
    unified_data["EITLEM"] = df_eit

    common_reactions = set(unified_data["DB-Kinetic"]['reaction_id'])
    for model in ["CataPro", "DLKcat", "UniKP", "EITLEM"]:
        common_reactions = common_reactions & set(unified_data[model]['reaction_id'])
    print(f"\n🔍 5个模型共有的reaction_id数量：{len(common_reactions)}")

    for model in unified_data:
        df = unified_data[model]
        df = df[df['reaction_id'].isin(common_reactions)]
        reaction_counts = df['reaction_id'].value_counts()
        valid_reactions = reaction_counts[reaction_counts >= 20].index
        df = df[df['reaction_id'].isin(valid_reactions)]
        unified_data[model] = df.reset_index(drop=True)
        print(f"✅ {model} 过滤完成：有效reaction数={len(valid_reactions)}，有效行数={len(df)}")
    return unified_data

def calculate_km_metrics(unified_data):
    metrics_dict = {}
    for model, df in unified_data.items():
        model_metrics = {}
        print(f"\n🔢 计算 {model} 的KM指标...")
        for reaction_id, group in df.groupby('reaction_id'):
            true_vals = group['true_log10_km'].values
            pred_vals = group['model_pred_log10'].values
            if len(np.unique(true_vals)) <= 1 or len(np.unique(pred_vals)) <= 1:
                continue
            scc, _ = stats.spearmanr(true_vals, pred_vals)
            scc = np.nan_to_num(scc)
            pcc, _ = stats.pearsonr(true_vals, pred_vals)
            pcc = np.nan_to_num(pcc)
            indices = list(range(len(group)))
            total_pairs = len(list(combinations(indices, 2)))
            correct_pairs = 0
            for i, j in combinations(indices, 2):
                true_i_higher = true_vals[i] > true_vals[j]
                pred_i_higher = pred_vals[i] > pred_vals[j]
                if true_i_higher == pred_i_higher:
                    correct_pairs += 1
            accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
            model_metrics[reaction_id] = {'scc': scc, 'pcc': pcc, 'accuracy': accuracy}
        metrics_dict[model] = model_metrics
        print(f"✅ {model} 计算完成：有效reaction数={len(model_metrics)}")
    return metrics_dict

def organize_km_metrics(metrics_dict):
    scc_data = {}
    pcc_data = {}
    acc_data = {}
    models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "EITLEM"]
    for model in models:
        model_metrics = metrics_dict.get(model, {})
        scc_data[model] = [v['scc'] for v in model_metrics.values()] if model_metrics else []
        pcc_data[model] = [v['pcc'] for v in model_metrics.values()] if model_metrics else []
        acc_data[model] = [v['accuracy'] for v in model_metrics.values()] if model_metrics else []
        print(f"📊 {model} | SCC={len(scc_data[model])}, PCC={len(pcc_data[model])}, ACC={len(acc_data[model])}")
    return scc_data, pcc_data, acc_data

# ======================================================================================
#
#   【3】kcat/Km 部分代码（my → DB-Kinetic）
#
# ======================================================================================
kcat_km_paths = {
    "CataPro": "/mnt/usb1/wmx/catapro/analyse/catapro_kcat_km/kcat-km_N20_act_pred.pkl",
    "DB-Kinetic": "/mnt/usb1/wmx/catapro/analyse/kcat_km/kcat-km_data_filtered_N20_with_pred.pkl",
    "DLKcat": "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/kcat_km/kcat_km_N20_1.pkl",
    "UniKP": "/mnt/usb1/wmx/unikp/analysis/kcat_km/kcat-km_N20.pkl",
    "EITLEM": "/mnt/usb1/wmx/eitlem/kcat/N20_KKM_final_pred.pkl"
}

def load_data(file_paths):
    data_dict = {}
    for model, path in file_paths.items():
        print(f"\n=== 加载 {model} 数据 ===")
        df = pd.read_pickle(path)
        df = df[(df['Km(M)'] > 0) & (df['reaction_id'].notnull())].reset_index(drop=True)
        print(f"  有效行数：{len(df)}")
        data_dict[model] = df
    return data_dict

def unify_scale(data_dict):
    unified = {}
    df = data_dict["CataPro"].copy()
    df['true_log10'] = df['log10_kcat_over_Km']
    df['pred_log10'] = df['pred_log10[kcat/Km(s^-1mM^-1)]']
    unified["CataPro"] = df

    df = data_dict["DB-Kinetic"].copy()
    df['true_log10'] = df['log10_kcat_over_Km']
    df['pred_log10'] = df['pred_fusion_log10[kcat/Km]']
    unified["DB-Kinetic"] = df

    df = data_dict["DLKcat"].copy()
    df['true_log10'] = df['log10_kcat_over_Km']
    df['pred_log10'] = df['Pred_kcat_km_log2'] / np.log2(10)
    unified["DLKcat"] = df

    df = data_dict["UniKP"].copy()
    df['true_log10'] = df['log10_kcat_over_Km']
    df['pred_log10'] = df['pred_log10_kcat_over_Km']
    unified["UniKP"] = df

    df_eit = data_dict["EITLEM"].copy()
    df_eit['true_log10'] = np.log10(df_eit['kcat(s^-1)'] / df_eit['Km(M)'])
    df_eit['pred_log10'] = df_eit['KKM_pred']
    unified["EITLEM"] = df_eit

    common = set(unified["DB-Kinetic"]['reaction_id'])
    for m in ["CataPro", "DLKcat", "UniKP", "EITLEM"]:
        common &= set(unified[m]['reaction_id'])
    print(f"\n共同 reaction 数量：{len(common)}")

    for m in unified:
        d = unified[m]
        d = d[d['reaction_id'].isin(common)]
        cnt = d['reaction_id'].value_counts()
        valid = cnt[cnt >= 20].index
        d = d[d['reaction_id'].isin(valid)]
        unified[m] = d.reset_index(drop=True)
        print(f"✅ {m} | 有效组：{len(valid)}，总行数：{len(d)}")
    return unified

def calc_metrics(unified):
    metrics = {}
    for model, df in unified.items():
        model_res = {}
        print(f"\n🔢 计算 {model} ...")
        for rid, g in df.groupby("reaction_id"):
            t = g['true_log10'].values
            p = g['pred_log10'].values
            if len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
                continue
            scc, _ = stats.spearmanr(t, p)
            pcc, _ = stats.pearsonr(t, p)
            scc = np.nan_to_num(scc)
            pcc = np.nan_to_num(pcc)
            correct = 0
            total = 0
            for i, j in combinations(range(len(g)), 2):
                if (t[i] > t[j]) == (p[i] > p[j]):
                    correct += 1
                total += 1
            acc = correct / total if total > 0 else 0
            model_res[rid] = {"scc": scc, "pcc": pcc, "accuracy": acc}
        metrics[model] = model_res
        print(f"✅ {model} 完成，有效数量：{len(model_res)}")
    return metrics

def arrange(metrics):
    models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "EITLEM"]
    scc, pcc, acc = {}, {}, {}
    for m in models:
        data = list(metrics[m].values())
        scc[m] = [x['scc'] for x in data]
        pcc[m] = [x['pcc'] for x in data]
        acc[m] = [x['accuracy'] for x in data]
        print(f"📊 {m} | SCC={len(scc[m])}, PCC={len(pcc[m])}, ACC={len(acc[m])}")
    return scc, pcc, acc

# ======================================================================================
#
#   绘图部分（已放大横轴模型字体）
#
# ======================================================================================
def plot_all_in_one_3x3(kcat_scc, kcat_pcc, kcat_acc,
                         km_scc, km_pcc, km_acc,
                         kcatkm_scc, kcatkm_pcc, kcatkm_acc):
    # 换更清晰字体 + 全局放大基础字号
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'path'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.titlepad'] = 10
    # 全局默认字号加大
    plt.rcParams['font.size'] = 12

    colors = {"DB-Kinetic": '#d62728', "CataPro": '#1f77b4', "DLKcat": '#ff7f0e', "UniKP": '#2ca02c', "EITLEM": "#9467bd"}
    models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "EITLEM"]
    jitter_strength = 0.12

    fig, axes = plt.subplots(3, 3, figsize=(21, 18))

    plot_info = [
        {"row": 0, "scc": kcat_scc, "pcc": kcat_pcc, "acc": kcat_acc,
         "title": r"$\mathrm{k_{cat} \ dataset}$", "labels": ["a", "b", "c"]},
        {"row": 1, "scc": km_scc, "pcc": km_pcc, "acc": km_acc,
         "title": r"$\mathrm{K_{m} \ dataset}$", "labels": ["d", "e", "f"]},
        {"row": 2, "scc": kcatkm_scc, "pcc": kcatkm_pcc, "acc": kcatkm_acc,
         "title": r"$\mathrm{k_{cat}/K_{m} \ dataset}$", "labels": ["g", "h", "i"]},
    ]

    def draw_subplot(ax, data, title, ylabel, ylim, label, zero=False):
        valid_data, valid_labels, curr_colors = [], [], []
        for m in models:
            if len(data[m]) > 0:
                valid_data.append(data[m])
                valid_labels.append(m)
                curr_colors.append(colors[m])

        box = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, showmeans=True,
                         widths=0.6, zorder=2,
                         medianprops={'color': 'white', 'linewidth': 2},
                         meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5},
                         flierprops={'marker': '.'})
        for patch, c in zip(box['boxes'], curr_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        for i, m in enumerate(valid_labels):
            y = data[m]
            x = np.random.normal(i + 1, jitter_strength, len(y))
            ax.scatter(x, y, c='black', alpha=0.4, s=8, zorder=3, linewidths=0)

        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if zero:
            ax.axhline(0, color='black', lw=1, alpha=0.2, zorder=1)

        # 横轴字体大幅放大 + 轻微旋转防重叠
        ax.tick_params(axis='x', labelsize=15)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        # ====================== 标签左移，不重合 ======================
        ax.text(-0.12, 1.01, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    for info in plot_info:
        row = info["row"]
        draw_subplot(axes[row, 0], info["scc"], info["title"], "Spearman Correlation", (-1.05, 1.05), info["labels"][0], zero=True)
        draw_subplot(axes[row, 1], info["pcc"], info["title"], "Pearson Correlation", (-1.05, 1.05), info["labels"][1], zero=True)
        draw_subplot(axes[row, 2], info["acc"], info["title"], "Pairwise Accuracy", (0, 1.05), info["labels"][2])

    plt.tight_layout(pad=3.5)
    plt.savefig(r"/mnt/usb1/wmx/catapro/analyse/all_3x3_figure_title_fixed.svg", bbox_inches='tight', dpi=300)
    plt.show()
    print("✅ 3×3大图已保存！abc标签位置已修正，不再重合")
# ======================================================================================
#
#   主执行流程
#
# ======================================================================================
if __name__ == "__main__":
    np.seterr(invalid='ignore', divide='ignore')

    # 1. kcat
    data_20 = load_kcat_data(file_config["N≥20"], 20)
    unified_20 = unify_log10_scale(data_20, 20)
    metrics_20 = calculate_metrics_per_reaction(unified_20)
    scc_20, pcc_20, acc_20 = organize_metrics(metrics_20)

    # 2. Km
    data_km = load_and_clean_km_data(km_file_paths)
    unified_km = unify_km_log10_scale(data_km)
    metrics_km = calculate_km_metrics(unified_km)
    scc_km, pcc_km, acc_km = organize_km_metrics(metrics_km)

    # 3. kcat/Km
    data_kcatkm = load_data(kcat_km_paths)
    unified_kcatkm = unify_scale(data_kcatkm)
    metrics_kcatkm = calc_metrics(unified_kcatkm)
    scc_kcatkm, pcc_kcatkm, acc_kcatkm = arrange(metrics_kcatkm)

    # 绘图
    plot_all_in_one_3x3(
        scc_20, pcc_20, acc_20,
        scc_km, pcc_km, acc_km,
        scc_kcatkm, pcc_kcatkm, acc_kcatkm
    )

# ===================== 统计汇总 =====================
def get_stat_summary(data_dict, name):
    models = ["DB-Kinetic", "CataPro", "DLKcat", "UniKP", "EITLEM"]
    res = []
    for m in models:
        arr = np.array(data_dict[m])
        if len(arr) == 0:
            res.append({
                "Dataset_Metric": name,
                "Model": m,
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
                "Q1": np.nan,
                "Q3": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan
            })
            continue
        res.append({
            "Dataset_Metric": name,
            "Model": m,
            "n": len(arr),
            "mean": round(np.mean(arr), 4),
            "median": round(np.median(arr), 4),
            "Q1": round(np.percentile(arr, 25), 4),
            "Q3": round(np.percentile(arr, 75), 4),
            "std": round(np.std(arr, ddof=1), 4),
            "min": round(np.min(arr), 4),
            "max": round(np.max(arr), 4)
        })
    return pd.DataFrame(res)