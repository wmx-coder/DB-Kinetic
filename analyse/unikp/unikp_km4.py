import numpy as np
import pandas as pd
import pickle
import os
import gc
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -------------------------- 核心函数：精简版KM十折验证 --------------------------
def KM_predict_10fold(feature, label, fold_labels, meta_data):
    """
    精简版KM十折交叉验证（仅保留核心逻辑+结果保存）
    参数说明：
        feature: 拼接后的特征矩阵（sbt_feat+ezy_feat）
        label: log10_km标签数组
        fold_labels: 样本的fold标签数组（numpy.ndarray）
        meta_data: 元数据字典（包含Sequence/Smiles等）
    """
    fold_ids = sorted(np.unique(fold_labels))
    print(f"开始KM十折交叉验证，fold编号：{fold_ids}")
    fold_metrics_summary = []
    save_dir = 'PreKM_new/10fold'
    os.makedirs(save_dir, exist_ok=True)

    # 初始化最终预测值数组（仅保留测试集预测值）
    final_predictions = np.full(len(feature), np.nan)

    for fold_idx, test_fold in enumerate(fold_ids):
        print(f"\n正在执行第 {fold_idx} 折验证，测试集fold：{test_fold}")
        # 划分训练/测试集索引
        train_index = np.where(fold_labels != test_fold)[0]
        test_index = np.where(fold_labels == test_fold)[0]

        # 提取训练/测试数据
        X_train, y_train = feature[train_index], label[train_index]
        X_test, y_test = feature[test_index], label[test_index]

        # 训练模型+预测
        model = ExtraTreesRegressor()
        model.fit(X_train, y_train)
        y_pred_all = model.predict(feature)
        final_predictions[test_index] = y_pred_all[test_index]  # 仅保留测试集预测值

        # 计算评估指标
        y_pred_test = y_pred_all[test_index]
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        pcc, pcc_pvalue = pearsonr(y_test, y_pred_test)
        scc, scc_pvalue = spearmanr(y_test, y_pred_test)

        # 打印当前折指标
        print(f"第 {fold_idx} 折评估指标（log10(KM)）：")
        print(f"R²={r2:.4f}, PCC={pcc:.4f}, SCC={scc:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        # 保存当前折指标
        fold_metrics_summary.append({
            'fold_index': fold_idx,
            'test_fold': int(test_fold),
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'pcc': pcc, 'pcc_pvalue': pcc_pvalue,
            'scc': scc, 'scc_pvalue': scc_pvalue,
            'test_sample_count': len(test_index)
        })

        # 保存当前折详细结果
        # 标记训练/测试集
        train_test_flag = np.zeros(len(feature), dtype=int)
        train_test_flag[test_index] = 1
        # 生成当前折全样本结果
        fold_res = pd.DataFrame({
            'Sequence': meta_data['Sequence'],
            'Smiles': meta_data['Smiles'],
            'EC': meta_data['EC'],
            'Organism': meta_data['Organism'],
            'Substrate': meta_data['Substrate'],
            'EnzymeType': meta_data['EnzymeType'],
            'true_log10_km': label,
            'pred_log10_km': y_pred_all,
            'fold': fold_labels,
            'is_test': train_test_flag
        })
        fold_res.to_excel(f'{save_dir}/第{fold_idx}折_测试集fold{test_fold}_all_samples.xlsx', index=False)
        # 生成当前折测试集结果（含指标）
        test_res = fold_res.iloc[test_index].copy()
        test_res['fold_r2'] = r2
        test_res['fold_pcc'] = pcc
        test_res['fold_rmse'] = rmse
        test_res['fold_mae'] = mae
        test_res.to_excel(f'{save_dir}/第{fold_idx}折_测试集fold{test_fold}_test_samples.xlsx', index=False)

    # 保存指标汇总表
    metrics_df = pd.DataFrame(fold_metrics_summary)
    # 添加平均值行
    avg_row = {
        'fold_index': '平均值', 'test_fold': '-',
        'mse': metrics_df['mse'].mean(), 'rmse': metrics_df['rmse'].mean(),
        'mae': metrics_df['mae'].mean(), 'r2': metrics_df['r2'].mean(),
        'pcc': metrics_df['pcc'].mean(), 'scc': metrics_df['scc'].mean(),
        'test_sample_count': metrics_df['test_sample_count'].sum(),
        'pcc_pvalue': '-', 'scc_pvalue': '-'
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)
    metrics_df.to_excel(f'{save_dir}/十折验证指标汇总(KM).xlsx', index=False)

    # 保存最终预测结果（仅测试集预测值+元数据）
    final_df = pd.DataFrame({
        'Sequence': meta_data['Sequence'],
        'Smiles': meta_data['Smiles'],
        'EC': meta_data['EC'],
        'Organism': meta_data['Organism'],
        'Substrate': meta_data['Substrate'],
        'EnzymeType': meta_data['EnzymeType'],
        'fold': fold_labels,
        'true_log10_km': label,
        'pred_log10_km': final_predictions,
        'true_km': np.where(np.isfinite(label), 10 ** label, np.nan),
        'pred_km': np.where(np.isfinite(final_predictions), 10 ** final_predictions, np.nan)
    })
    final_df.to_excel(f'{save_dir}/最终整合数据集_仅测试集预测值(KM).xlsx', index=False)
    final_df.to_csv(f'{save_dir}/最终整合数据集_仅测试集预测值(KM).csv', index=False, encoding='utf-8')

    # 检查未覆盖样本
    nan_count = final_df['pred_log10_km'].isna().sum()
    if nan_count > 0:
        print(f"\n警告：{nan_count} 个样本无预测值（fold划分异常）")
    else:
        print("\n✅ 所有样本均有测试集预测值")

    print(f"\n✅ KM十折验证完成！平均R²={metrics_df['r2'].mean():.4f}")
    return metrics_df, final_predictions


# -------------------------- 主函数：加载数据+过滤无效样本+执行验证 --------------------------
if __name__ == '__main__':
    # -------------------------- 配置路径 --------------------------
    PROCESSED_PKL = "PreKm_new/10fold/km_data_with_id_log_processed.pkl"
    SMILES_FEAT_PKL = "PreKm_new/10fold/km_smiles_id_feat.pkl"

    # -------------------------- 加载数据集 --------------------------
    print(f"1. 加载KM数据集：{PROCESSED_PKL}")
    df = pd.read_pickle(PROCESSED_PKL)
    df.columns = [col.strip() for col in df.columns]  # 清理列名空格
    if 'id' in df.columns and 'smiles_id' not in df.columns:
        df.rename(columns={'id': 'smiles_id'}, inplace=True)

    # -------------------------- 核心过滤：只保留1024维ezy_feat样本 --------------------------
    print(f"\n2. 过滤ezy_feat无效样本（仅保留1024维numpy数组）")
    original_sample_count = len(df)
    # 过滤条件：ezy_feat是numpy数组 且 维度为1024
    valid_ezy_mask = df['ezy_feat'].apply(lambda x: isinstance(x, np.ndarray) and x.shape == (1024,))
    df = df[valid_ezy_mask].reset_index(drop=True)
    filtered_sample_count = len(df)
    removed_count = original_sample_count - filtered_sample_count
    print(f"   原始样本数：{original_sample_count}")
    print(f"   过滤后样本数：{filtered_sample_count}")
    print(f"   过滤掉无效样本数：{removed_count}")

    # -------------------------- 验证核心列 --------------------------
    required_cols = ['ezy_feat', 'smiles_id', 'fold', 'log10_km', 'Sequence', 'Smiles', 'EC', 'EnzymeType', 'Organism',
                     'Substrate']
    for col in required_cols:
        assert col in df.columns, f"缺少核心列：{col}"

    # -------------------------- 加载SMILES特征 --------------------------
    print(f"\n3. 加载SMILES特征：{SMILES_FEAT_PKL}")
    with open(SMILES_FEAT_PKL, 'rb') as f:
        smiles_data = pickle.load(f)
    id2feat = smiles_data['id_to_smiles_feat']

    # 安全加载SMILES特征（防止smiles_id不存在）
    smiles_feat = []
    for sid in df['smiles_id']:
        if sid in id2feat:
            smiles_feat.append(id2feat[sid])
        else:
            smiles_feat.append(np.zeros(1024))  # 缺失特征补零
    smiles_feat = np.array(smiles_feat)  # 此时可安全转换

    # -------------------------- 提取酶特征 --------------------------
    print(f"4. 提取ezy_feat酶特征")
    # 过滤后所有ezy_feat都是1024维，可安全转换
    ezy_feat = np.array(df['ezy_feat'].tolist())
    print(f"   ezy_feat形状：{ezy_feat.shape}")
    print(f"   smiles_feat形状：{smiles_feat.shape}")

    # -------------------------- 特征拼接+过滤无效标签 --------------------------
    print(f"\n5. 特征拼接+过滤标签异常样本")
    # 拼接特征
    feature = np.concatenate([smiles_feat, ezy_feat], axis=1)
    # 过滤无效标签样本（log10_km异常 或 SMILES含非法字符）
    valid_mask = (df['log10_km'] > -10000000000) & (~df['Smiles'].str.contains('\.'))
    feature_valid = feature[valid_mask]
    label_valid = df.loc[valid_mask, 'log10_km'].values
    fold_valid = df.loc[valid_mask, 'fold'].values

    # 提取元数据（用于保存结果）
    meta_data = {
        'Sequence': df.loc[valid_mask, 'Sequence'].tolist(),
        'Smiles': df.loc[valid_mask, 'Smiles'].tolist(),
        'EC': df.loc[valid_mask, 'EC'].tolist(),
        'Organism': df.loc[valid_mask, 'Organism'].tolist(),
        'Substrate': df.loc[valid_mask, 'Substrate'].tolist(),
        'EnzymeType': df.loc[valid_mask, 'EnzymeType'].tolist()
    }

    print(f"   过滤后有效样本数：{len(label_valid)}")
    print(f"   最终特征维度：{feature_valid.shape}")

    # -------------------------- 执行十折交叉验证 --------------------------
    print(f"\n6. 开始执行KM十折交叉验证")
    metrics_summary, final_pred = KM_predict_10fold(
        feature=feature_valid,
        label=label_valid,
        fold_labels=fold_valid,
        meta_data=meta_data
    )

    # -------------------------- 保存最终指标（可选） --------------------------
    with open('PreKM_new/KM_10fold_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_summary, f)
    print("\n🎉 所有流程完成！结果已保存至 PreKM_new/10fold 目录")