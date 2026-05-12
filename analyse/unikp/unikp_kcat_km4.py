import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def KcatKm_predict_10fold(feature, label, fold_labels, meta_data):
    """
    增强版kcat/km十折交叉验证函数（新增元数据保存，对齐KM代码逻辑）
    参数说明：
        feature: 拼接后的特征矩阵（SMILES+ezy_feat）
        label: log10_kcat_over_Km标签数组
        fold_labels: 样本的fold标签数组（numpy.ndarray）
        meta_data: 元数据字典（包含Sequence/Smiles等）
    输出：指标汇总+最终预测值
    """
    # 修复：numpy数组用np.unique()获取唯一值，再排序
    fold_ids = sorted(np.unique(fold_labels))
    print(f"开始十折交叉验证（kcat/km），fold编号：{fold_ids}")
    fold_metrics_summary = []

    # 创建保存目录（对齐KM代码的目录结构）
    save_dir = 'PreKcat_km/10fold'
    os.makedirs(save_dir, exist_ok=True)

    # 初始化最终预测值数组
    final_predictions = np.full(len(feature), np.nan)

    for fold_idx, test_fold in enumerate(fold_ids):
        print(f"\n正在执行第 {fold_idx} 折验证，测试集fold：{test_fold}")
        # 划分训练/测试集索引（numpy数组用==/!=判断）
        train_index = np.where(fold_labels != test_fold)[0]
        test_index = np.where(fold_labels == test_fold)[0]

        # 提取训练/测试数据（仅核心特征+标签）
        X_train, y_train = feature[train_index], label[train_index]
        X_test, y_test = feature[test_index], label[test_index]

        # 训练模型+预测（仅核心逻辑）
        model = ExtraTreesRegressor()
        model.fit(X_train, y_train)
        y_pred_all = model.predict(feature)
        final_predictions[test_index] = y_pred_all[test_index]  # 仅保留测试集预测值

        # 计算评估指标（仅核心指标，无冗余）
        y_pred_test = y_pred_all[test_index]
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        pcc, pcc_pvalue = pearsonr(y_test, y_pred_test)
        scc, scc_pvalue = spearmanr(y_test, y_pred_test)

        # 打印当前折指标
        print(f"第 {fold_idx} 折评估指标（log10(kcat/km)）：")
        print(f"R²={r2:.4f}, PCC={pcc:.4f}, SCC={scc:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        # 保存当前折指标（仅指标，无元数据）
        fold_metrics_summary.append({
            'fold_index': fold_idx,
            'test_fold': int(test_fold),
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'pcc': pcc, 'pcc_pvalue': pcc_pvalue,
            'scc': scc, 'scc_pvalue': scc_pvalue,
            'test_sample_count': len(test_index)
        })

        # ========== 新增：保存每折全样本结果（对齐KM代码） ==========
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
            'true_log10_kcat_over_Km': label,
            'pred_log10_kcat_over_Km': y_pred_all,
            'fold': fold_labels,
            'is_test': train_test_flag
        })
        fold_res.to_excel(f'{save_dir}/第{fold_idx}折_测试集fold{test_fold}_all_samples.xlsx', index=False)

        # ========== 新增：保存当前折测试集结果（含指标，对齐KM代码） ==========
        test_res = fold_res.iloc[test_index].copy()
        test_res['fold_r2'] = r2
        test_res['fold_pcc'] = pcc
        test_res['fold_rmse'] = rmse
        test_res['fold_mae'] = mae
        test_res.to_excel(f'{save_dir}/第{fold_idx}折_测试集fold{test_fold}_test_samples.xlsx', index=False)

    # 保存指标汇总表（仅指标，无任何元数据）
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
    metrics_df.to_excel(f'{save_dir}/十折验证指标汇总(kcat_km).xlsx', index=False)

    # ========== 新增：保存最终整合数据集（含元数据+Excel+CSV，对齐KM代码） ==========
    final_df = pd.DataFrame({
        'Sequence': meta_data['Sequence'],
        'Smiles': meta_data['Smiles'],
        'EC': meta_data['EC'],
        'Organism': meta_data['Organism'],
        'Substrate': meta_data['Substrate'],
        'EnzymeType': meta_data['EnzymeType'],
        'fold': fold_labels,
        'true_log10_kcat_over_Km': label,
        'pred_log10_kcat_over_Km': final_predictions,
        # 可选：还原kcat/Km原始值（非log10）
        'true_kcat_over_Km': np.where(np.isfinite(label), 10 ** label, np.nan),
        'pred_kcat_over_Km': np.where(np.isfinite(final_predictions), 10 ** final_predictions, np.nan)
    })
    # 保存Excel和CSV格式（方便后续匹配）
    final_df.to_excel(f'{save_dir}/最终整合数据集_仅测试集预测值(kcat_km).xlsx', index=False)
    final_df.to_csv(f'{save_dir}/最终整合数据集_仅测试集预测值(kcat_km).csv', index=False, encoding='utf-8')

    # 保留原有基础预测值文件（兼容你的历史代码）
    pred_df = pd.DataFrame({
        'true_log10_kcat_over_Km': label,
        'pred_log10_kcat_over_Km': final_predictions,
        'fold': fold_labels
    })
    pred_df.to_excel(f'{save_dir}/预测结果(kcat_km).xlsx', index=False)

    # 检查未覆盖样本
    nan_count = final_df['pred_log10_kcat_over_Km'].isna().sum()
    if nan_count > 0:
        print(f"\n警告：{nan_count} 个样本无预测值（fold划分异常）")
    else:
        print("\n✅ 所有样本均有测试集预测值")

    print(f"\n✅ kcat/km十折验证完成！平均R²={metrics_df['r2'].mean():.4f}")
    return metrics_df, final_predictions  # 仅返回指标+预测值，无元数据


if __name__ == '__main__':
    # -------------------------- 配置路径 --------------------------
    PROCESSED_PKL = "PreKcat_km/10fold/data_with_id_log_processed.pkl"
    SMILES_FEAT_PKL = "PreKcat_km/10fold/smiles_id_feat.pkl"

    # -------------------------- 加载数据集（提取元数据，不再丢弃） --------------------------
    print(f"1. 加载kcat/km数据集：{PROCESSED_PKL}")
    df = pd.read_pickle(PROCESSED_PKL)
    df.columns = [col.strip() for col in df.columns]  # 清理列名空格
    if 'id' in df.columns and 'smiles_id' not in df.columns:
        df.rename(columns={'id': 'smiles_id'}, inplace=True)

    # 验证核心必要列（包含元数据列）
    required_cols = ['ezy_feat', 'smiles_id', 'fold', 'log10_kcat_over_Km',
                     'Sequence', 'Smiles', 'EC', 'EnzymeType', 'Organism', 'Substrate']
    for col in required_cols:
        assert col in df.columns, f"缺少核心列：{col}"

    # -------------------------- 加载SMILES特征 --------------------------
    print(f"2. 加载SMILES特征：{SMILES_FEAT_PKL}")
    with open(SMILES_FEAT_PKL, 'rb') as f:
        smiles_data = pickle.load(f)
    id2feat = smiles_data['id_to_smiles_feat']
    smiles_feat = np.array([id2feat[sid] for sid in df['smiles_id']])

    # -------------------------- 提取酶特征 --------------------------
    print(f"3. 提取ezy_feat酶特征")
    ezy_feat = np.array(df['ezy_feat'].tolist())

    # -------------------------- 特征拼接+过滤（保留元数据） --------------------------
    # 拼接特征
    feature = np.concatenate([smiles_feat, ezy_feat], axis=1)
    # 过滤无效样本（仅基于标签和SMILES有效性）
    valid_mask = (df['log10_kcat_over_Km'] > -10000000000) & (~df['Smiles'].str.contains('\.'))
    # 保留核心数据+元数据
    feature_valid = feature[valid_mask]
    label_valid = df.loc[valid_mask, 'log10_kcat_over_Km'].values
    fold_valid = df.loc[valid_mask, 'fold'].values  # 转为numpy数组

    # ========== 新增：提取元数据（对齐KM代码） ==========
    meta_data = {
        'Sequence': df.loc[valid_mask, 'Sequence'].tolist(),
        'Smiles': df.loc[valid_mask, 'Smiles'].tolist(),
        'EC': df.loc[valid_mask, 'EC'].tolist(),
        'Organism': df.loc[valid_mask, 'Organism'].tolist(),
        'Substrate': df.loc[valid_mask, 'Substrate'].tolist(),
        'EnzymeType': df.loc[valid_mask, 'EnzymeType'].tolist()
    }

    print(f"4. 过滤后有效样本数：{len(label_valid)}")
    print(f"   特征维度：{feature_valid.shape}")

    # -------------------------- 执行增强版十折验证（新增meta_data入参） --------------------------
    metrics_summary, final_pred = KcatKm_predict_10fold(
        feature=feature_valid,
        label=label_valid,
        fold_labels=fold_valid,
        meta_data=meta_data  # 传入元数据
    )

    # ========== 新增：保存指标PKL文件（对齐KM代码） ==========
    with open('PreKcat_km/KcatKm_10fold_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_summary, f)
    print("\n🎉 kcat/km十折验证所有流程完成！结果已保存至 PreKcat_km/10fold 目录")