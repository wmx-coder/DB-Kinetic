import pandas as pd
import numpy as np
import torch as th
import os
from torch.utils.data import DataLoader, Dataset
from model2 import Model_Regression
from argparse import RawDescriptionHelpFormatter
import argparse
import logging

# 抑制冗余日志
logging.basicConfig(level=logging.ERROR)


# ------------------------- 数据集类（适配预测场景，仅加载特征无标签） -------------------------
class Mydatasets_Infer(Dataset):
    def __init__(self, ezy_feats, sbt_feats):
        self.ezy_feats = ezy_feats  # list of (seq_len, 1024) ndarray
        self.sbt_feats = sbt_feats  # (N, 935) ndarray

    def __getitem__(self, idx):
        ezy = th.from_numpy(self.ezy_feats[idx]).float()
        sbt = th.from_numpy(self.sbt_feats[idx]).float()
        return ezy, sbt

    def __len__(self):
        return len(self.ezy_feats)


# ------------------------- 数据加载与预处理（适配预测） -------------------------
def load_infer_data(fpath):
    """加载预测数据集，提取特征和fold信息"""
    # 读取完整数据集（保留所有列）
    full_df = pd.read_pickle(fpath).copy()
    print(f"成功加载预测数据集：{fpath}，共 {len(full_df)} 条数据")

    # 校验必要列
    required_cols = ["new_ezy_feat", "sbt_feat", "fold"]
    missing_cols = [col for col in required_cols if col not in full_df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必要列：{missing_cols}")

    # 处理酶特征（list of (seq_len, 1024) ndarray）
    ezy_feats_list = []
    invalid_ezy_idx = []
    for idx, feat in enumerate(full_df["new_ezy_feat"]):
        if isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024:
            ezy_feats_list.append(feat)
        else:
            invalid_ezy_idx.append(idx)
            print(f"警告：索引{idx}的new_ezy_feat格式错误，将填充NaN")
            ezy_feats_list.append(np.zeros((1, 1024)))  # 占位

    # 处理底物特征（(N, 935) ndarray）
    sbt_feats = []
    invalid_sbt_idx = []
    for idx, feat in enumerate(full_df["sbt_feat"]):
        if isinstance(feat, np.ndarray) and feat.ndim == 1 and len(feat) == 935:
            sbt_feats.append(feat)
        else:
            invalid_sbt_idx.append(idx)
            print(f"警告：索引{idx}的sbt_feat格式错误，将填充NaN")
            sbt_feats.append(np.zeros(935))  # 占位
    sbt_feats = np.array(sbt_feats)

    # 处理fold列（转为整数）
    full_df["fold"] = full_df["fold"].astype(int)
    folds = full_df["fold"].values

    # 记录无效索引
    invalid_idx = list(set(invalid_ezy_idx + invalid_sbt_idx))
    print(f"有效特征样本数：{len(ezy_feats_list) - len(invalid_idx)}，无效样本数：{len(invalid_idx)}")

    return full_df, ezy_feats_list, sbt_feats, folds, invalid_idx


def collate_fn_infer(batch):
    """预测场景的collate_fn（同训练逻辑，仅无label）"""
    ezy_list, sbt_list = zip(*batch)
    batch_size = len(ezy_list)
    max_seq_len = max(ezy.shape[0] for ezy in ezy_list)

    # 酶特征padding + mask
    ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
    enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
    for i, ezy in enumerate(ezy_list):
        seq_len = ezy.shape[0]
        ezy_padded[i, :seq_len, :] = ezy
        enzyme_mask[i, :, :seq_len] = 1

    # 底物特征tensor
    sbt_tensor = th.stack(sbt_list, dim=0)

    return ezy_padded, sbt_tensor, enzyme_mask


# ------------------------- 预测核心函数 -------------------------
def infer_km_by_fold(
    full_df, ezy_feats_list, sbt_feats, folds, invalid_idx,
    model_save_dir, device, batch_size=8
):
    """按fold加载对应模型，批量预测Km值"""
    # 初始化底物压缩层（同训练逻辑）
    sbt_compress = th.nn.Linear(935, 256).to(device)

    # 初始化预测结果数组（默认NaN）
    pred_km = np.full(len(full_df), np.nan, dtype=np.float32)

    # 按fold分组处理
    unique_folds = np.unique(folds)
    for fold in unique_folds:
        print(f"\n=== 处理Fold {fold} ===")
        # 筛选当前fold的样本索引
        fold_mask = folds == fold
        fold_idx = np.where(fold_mask)[0].tolist()
        if not fold_idx:
            print(f"Fold {fold} 无样本，跳过")
            continue

        # 筛选当前fold的特征
        fold_ezy_feats = [ezy_feats_list[i] for i in fold_idx]
        fold_sbt_feats = sbt_feats[fold_mask]

        # 创建预测数据集和dataloader
        infer_dataset = Mydatasets_Infer(fold_ezy_feats, fold_sbt_feats)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=collate_fn_infer
        )

        # 加载当前fold的最佳模型
        model_ckpt_path = os.path.join(model_save_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(model_ckpt_path):
            print(f"警告：Fold {fold} 的模型文件不存在({model_ckpt_path})，该fold样本预测值为NaN")
            continue

        # 加载模型和压缩层参数
        ckpt = th.load(model_ckpt_path, map_location=device)
        model = Model_Regression().to(device)
        model.load_state_dict(ckpt["model_state"])
        sbt_compress.load_state_dict(ckpt["compress_state"])

        # 模型设为评估模式
        model.eval()
        sbt_compress.eval()

        # 批量预测
        fold_pred = []
        with th.no_grad():
            for batch in infer_loader:
                ezy_padded, sbt_tensor, enzyme_mask = [x.to(device) for x in batch]
                # 底物特征压缩（同训练逻辑）
                reactions = sbt_compress(sbt_tensor).unsqueeze(1)
                # 预测
                pred, _ = model(reactions, ezy_padded, enzyme_mask=enzyme_mask)
                pred = pred.squeeze(-1).cpu().numpy()
                fold_pred.extend(pred.tolist())

        # 将预测结果赋值到对应位置
        pred_km[fold_idx] = fold_pred
        print(f"Fold {fold} 预测完成，样本数：{len(fold_idx)}")

    # 无效样本保持NaN
    pred_km[invalid_idx] = np.nan
    return pred_km


# ------------------------- 主函数 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Km Model Inference",
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-infer_fpath", type=str,
                        default="/mnt/usb3/wmx/km/case/km_final_valid_samples.pkl",
                        help="Path to inference dataset (pkl)")
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/km/km_model2_1",
                        help="Directory of trained best models")
    parser.add_argument("-batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
    parser.add_argument("-out_fpath", type=str,
                        default="/mnt/usb3/wmx/km/case/predict.pkl",
                        help="Output path with prediction")
    args = parser.parse_args()

    # 1. 加载预测数据
    full_df, ezy_feats_list, sbt_feats, folds, invalid_idx = load_infer_data(args.infer_fpath)

    # 2. 按fold预测Km值
    pred_km = infer_km_by_fold(
        full_df=full_df,
        ezy_feats_list=ezy_feats_list,
        sbt_feats=sbt_feats,
        folds=folds,
        invalid_idx=invalid_idx,
        model_save_dir=args.model_save_dir,
        device=args.device,
        batch_size=args.batch_size
    )

    # 3. 新增预测列到原始数据集
    full_df["pred_log10[Km(mM)]"] = pred_km
    print(f"\n预测完成！新增列：pred_log10[Km(mM)]")
    print(f"  - 有效预测数：{np.count_nonzero(~np.isnan(pred_km))}")
    print(f"  - 无效预测数（NaN）：{np.count_nonzero(np.isnan(pred_km))}")

    # 4. 保存结果
    # 4.1 保存完整pkl文件（保留所有列，包括特征数组）
    full_df.to_pickle(args.out_fpath)
    print(f"\n完整结果已保存至：{args.out_fpath}")

    # 4.2 保存CSV文件（跳过数组列，方便查看）
    csv_out_fpath = args.out_fpath.replace(".pkl", ".csv")
    non_array_cols = [col for col in full_df.columns if not isinstance(full_df[col].iloc[0], np.ndarray)]
    full_df[non_array_cols].to_csv(csv_out_fpath, index=False, encoding="utf-8")
    print(f"CSV预览文件已保存至：{csv_out_fpath}（仅含非数组列）")

    # 5. 打印预测统计
    print(f"\n=== 预测结果统计 ===")
    print(f"原始数据集总行数：{len(full_df)}")
    print(f"成功预测行数：{np.count_nonzero(~np.isnan(pred_km))}")
    print(f"失败预测行数（NaN）：{np.count_nonzero(np.isnan(pred_km))}")
    print(f"预测成功率：{np.count_nonzero(~np.isnan(pred_km)) / len(full_df) * 100:.2f}%")