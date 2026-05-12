#!/usr/bin/python
# coding: utf-8
import os
import json
import pickle
import pandas as pd
import numpy as np

# ====================== 配置项（Km专用，已替换所有路径和字段） ======================
# 1. Km小数据集路径（PKL）
SMALL_DATA_PATH = "/mnt/usb3/wmx/analyse/km-data_filtered_N30_matched.pkl"
# 2. Km全量数据集路径（JSON）
FULL_DATA_PATH = "/mnt/usb1/wmx/dlkcat/Data/database/Km_combination_0918.json"
# 3. Km全量预测结果路径（TSV）
PREDICTION_PATH = "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/km/km_prediction.tsv"
# 4. 输出匹配结果路径
OUTPUT_PATH = "./km_N30_matched_result.pkl"


# ====================== 核心匹配逻辑（Km专用，对齐原逻辑） ======================
def main():
    # ---------------------- 1. 加载数据 ----------------------
    print("=" * 60)
    print("1. 加载Km相关数据...")
    # 加载Km小数据集（PKL）
    df_small = pd.read_pickle(SMALL_DATA_PATH)
    print(f"✅ Km小数据集加载成功：{len(df_small)} 条样本")
    print(f"   小数据集列数：{len(df_small.columns)} 列")
    print(f"   小数据集字段预览：{list(df_small.columns)[:10]}")  # 预览前10个字段

    # 加载Km全量数据集（JSON）
    with open(FULL_DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    df_full = pd.DataFrame(full_data)
    print(f"✅ Km全量数据集加载成功：{len(df_full)} 条样本")

    # 加载Km全量预测结果（TSV）
    df_pred = pd.read_csv(PREDICTION_PATH, sep='\t')
    print(f"✅ Km全量预测结果加载成功：{len(df_pred)} 条样本")
    print(f"   预测结果字段：{list(df_pred.columns)}")

    # ---------------------- 2. 数据预处理（统一标识字段） ----------------------
    print("\n" + "=" * 60)
    print("2. 数据预处理（生成唯一匹配键）...")
    # 给全量数据集添加索引（对应预测结果的Sample_Index）
    df_full['Sample_Index'] = df_full.index

    # 生成唯一标识（Smiles + || + Sequence）：去除空格/换行，统一格式
    def generate_unique_key(df):
        # 确保字段类型为字符串，处理缺失值
        df['Smiles'] = df['Smiles'].fillna('').astype(str).str.strip()
        df['Sequence'] = df['Sequence'].fillna('').astype(str).str.strip()
        # 生成唯一键（核心匹配依据）
        df['Unique_Key'] = df['Smiles'] + "||" + df['Sequence']
        return df

    # 对小数据集和全量数据集生成唯一键
    df_small = generate_unique_key(df_small)
    df_full = generate_unique_key(df_full)
    print(f"✅ 唯一标识生成完成：Smiles + || + Sequence")

    # ---------------------- 3. 匹配：Km小数据集 ↔ 全量数据集 ↔ 预测结果 ----------------------
    print("\n" + "=" * 60)
    print("3. 开始精准匹配...")
    # 步骤1：小数据集关联全量数据集的Sample_Index和真实Km值
    df_match = pd.merge(
        df_small,  # Km小数据集（左表）
        df_full[['Unique_Key', 'Sample_Index', 'Value']],  # 全量数据集匹配字段（Value=Km真实值）
        on='Unique_Key',  # 按唯一键匹配
        how='left',  # 保留小数据集所有行
        suffixes=('', '_full')
    )

    # 步骤2：关联全量预测结果（按Sample_Index匹配）
    # 适配Km预测结果的字段名（兼容Pred_Km_log2/Pred_km_log2等命名）
    pred_log2_col = None
    pred_original_col = None
    # 自动识别Km预测字段
    for col in df_pred.columns:
        if 'Pred_Km_log2' in col or 'Pred_km_log2' in col or 'Pred_km_log2' in col:
            pred_log2_col = col
        if 'Pred_Km_original' in col or 'Pred_km_original' in col:
            pred_original_col = col
    # 兜底（确保字段存在）
    pred_log2_col = pred_log2_col if pred_log2_col else 'Pred_Km_log2'
    pred_original_col = pred_original_col if pred_original_col else 'Pred_Km_original'

    df_match = pd.merge(
        df_match,
        df_pred[['Sample_Index', pred_log2_col, pred_original_col, 'Model_Path']],
        on='Sample_Index',
        how='left'
    )

    # ---------------------- 4. 数据校验 & 结果整理 ----------------------
    print("\n" + "=" * 60)
    print("4. 匹配结果校验...")
    # 统计匹配成功/失败数
    match_success = df_match['Sample_Index'].notna().sum()
    match_fail = df_match['Sample_Index'].isna().sum()
    total_small = len(df_match)

    print(f"📊 Km匹配统计：")
    print(f"   Km小数据集总样本：{total_small} 条")
    print(f"   ✅ 匹配成功：{match_success} 条 ({match_success / total_small * 100:.2f}%)")
    print(f"   ❌ 匹配失败：{match_fail} 条 ({match_fail / total_small * 100:.2f}%)")

    # 输出匹配失败的样本（便于排查）
    if match_fail > 0:
        df_fail = df_match[df_match['Sample_Index'].isna()][['Smiles', 'Sequence', 'Unique_Key']].copy()
        # 截取长字符串便于查看
        df_fail['Smiles'] = df_fail['Smiles'].str[:50].str.ljust(50) + "..."
        df_fail['Sequence'] = df_fail['Sequence'].str[:50].str.ljust(50) + "..."
        # 保存失败样本
        fail_csv_path = "./km_match_failed_samples.csv"
        df_fail.to_csv(fail_csv_path, index=False, encoding='utf-8')
        print(f"⚠️  匹配失败样本已保存至：{fail_csv_path}")
        print(f"   失败样本预览（前3条）：")
        print(df_fail.head(3).to_string(index=False))

    # ---------------------- 5. 清理字段 & 标准化命名 & 保存结果 ----------------------
    print("\n" + "=" * 60)
    print("5. 保存最终匹配结果...")
    # 清理临时字段（保留小数据集所有原始字段 + 预测相关字段）
    drop_cols = ['Unique_Key']  # 删除临时生成的唯一键
    df_result = df_match.drop(columns=drop_cols, errors='ignore')

    # 标准化字段命名（统一为Km语义，更直观）
    rename_map = {
        pred_log2_col: 'Pred_Km_log2',  # Km预测值（log2）
        pred_original_col: 'Pred_Km_original',  # Km预测值（原始值）
        'Value': 'True_Km_original',  # Km真实值（全量JSON）
        'Model_Path': 'Used_Km_Model_Path',  # 使用的Km模型路径
        'Sample_Index': 'Km_Full_Data_Index'  # Km全量数据中的索引
    }
    # 仅重命名存在的字段（避免KeyError）
    rename_map = {k: v for k, v in rename_map.items() if k in df_result.columns}
    df_result.rename(columns=rename_map, inplace=True)

    # 保存结果（PKL + CSV，双重保障）
    # PKL格式（保留所有数据类型，便于后续分析）
    df_result.to_pickle(OUTPUT_PATH)
    # CSV格式（便于Excel/表格工具查看）
    csv_path = OUTPUT_PATH.replace('.pkl', '.csv')
    df_result.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"✅ Km匹配结果已保存：")
    print(f"   - PKL格式（完整数据）：{OUTPUT_PATH}")
    print(f"   - CSV格式（便于查看）：{csv_path}")

    # 最终结果预览
    print("\n" + "=" * 60)
    print("6. Km匹配结果预览（前3行核心字段）：")
    # 定义核心字段（兼容不同命名）
    core_cols = [
        'Smiles', 'Sequence', 'Km(M)', 'fold',
        'True_Km_original', 'Pred_Km_log2', 'Pred_Km_original', 'Used_Km_Model_Path'
    ]
    # 过滤存在的列（避免KeyError）
    core_cols = [col for col in core_cols if col in df_result.columns]
    df_preview = df_result[core_cols].head(3).copy()

    # 格式化预览（截取长字符串 + 数值保留4位小数）
    if 'Smiles' in df_preview.columns:
        df_preview['Smiles'] = df_preview['Smiles'].str[:30] + "..."
    if 'Sequence' in df_preview.columns:
        df_preview['Sequence'] = df_preview['Sequence'].str[:30] + "..."
    if 'Used_Km_Model_Path' in df_preview.columns:
        df_preview['Used_Km_Model_Path'] = df_preview['Used_Km_Model_Path'].str[-50:] + "..."

    # 数值字段格式化
    numeric_cols = ['Km(M)', 'True_Km_original', 'Pred_Km_log2', 'Pred_Km_original']
    for col in numeric_cols:
        if col in df_preview.columns:
            df_preview[col] = df_preview[col].round(4)

    print(df_preview.to_string(index=False))


if __name__ == '__main__':
    main()
    print("\n" + "=" * 60)
    print("🎉 Km数据集匹配完成！所有操作已执行完毕。")