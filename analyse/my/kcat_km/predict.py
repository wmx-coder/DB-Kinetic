import pandas as pd
import numpy as np
import torch as th
import os
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
import logging

# 抑制冗余日志
logging.basicConfig(level=logging.ERROR)

# 导入训练代码中的核心依赖
from util_check import RMSELoss, rmse_loss, evaluate
from model_kcat_km_1 import Model_Regression, KcatOriginalActivityModel, ActivityModel_Freeze as ActivityModel


# ------------------------- 数据集类（适配预测场景，仅加载特征无标签） -------------------------
class Mydatasets_Infer(Dataset):
    def __init__(self, ezy_feats, sbt_feats):
        self.ezy_feats = ezy_feats  # list of (seq_len, 1024) ndarray
        self.sbt_feats = sbt_feats  # (n_samples, 935) ndarray

    def __getitem__(self, idx):
        ezy = th.from_numpy(self.ezy_feats[idx]).float()
        sbt = th.from_numpy(self.sbt_feats[idx]).float()
        return ezy, sbt

    def __len__(self):
        return len(self.ezy_feats)


# ------------------------- 数据加载与预处理（适配kcat-km预测） -------------------------
def load_kcat_km_infer_data(fpath):
    """加载kcat-km预测数据集，提取特征、fold信息，保留原始所有列"""
    # 读取完整数据集
    full_df = pd.read_pickle(fpath).copy()
    print(f"成功加载kcat-km预测数据集：{fpath}，共 {len(full_df)} 条数据")

    # 校验必要列
    required_cols = ["new_ezy_feat", "sbt_feat", "fold"]
    missing_cols = [col for col in required_cols if col not in full_df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必要列：{missing_cols}")

    # 1. 处理酶特征（list of (seq_len, 1024) ndarray）
    ezy_feats_list = []
    invalid_ezy_idx = []
    for idx, feat in enumerate(full_df["new_ezy_feat"]):
        if isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024:
            ezy_feats_list.append(feat)
        else:
            invalid_ezy_idx.append(idx)
            print(f"警告：索引{idx}的new_ezy_feat格式错误，预测值为NaN")
            ezy_feats_list.append(np.zeros((1, 1024)))  # 占位

    # 2. 处理底物特征（(n_samples, 935) ndarray）
    sbt_feats = []
    invalid_sbt_idx = []
    for idx, feat in enumerate(full_df["sbt_feat"]):
        if isinstance(feat, np.ndarray) and feat.ndim == 1 and len(feat) == 935:
            sbt_feats.append(feat)
        else:
            invalid_sbt_idx.append(idx)
            print(f"警告：索引{idx}的sbt_feat格式错误，预测值为NaN")
            sbt_feats.append(np.zeros(935))  # 占位
    sbt_feats = np.array(sbt_feats)

    # 3. 处理fold列（转为整数）
    full_df["fold"] = full_df["fold"].astype(int)
    folds = full_df["fold"].values

    # 4. 记录无效索引（特征格式错误）
    invalid_idx = list(set(invalid_ezy_idx + invalid_sbt_idx))
    print(f"有效特征样本数：{len(ezy_feats_list) - len(invalid_idx)}，无效样本数：{len(invalid_idx)}")

    return full_df, ezy_feats_list, sbt_feats, folds, invalid_idx


def collate_fn_infer(batch):
    """预测场景的collate_fn（同训练逻辑，仅无label）"""
    ezy_list, sbt_list = zip(*batch)
    batch_size = len(ezy_list)
    max_seq_len = max(ezy.shape[0] for ezy in ezy_list)

    # 酶特征padding + 掩码
    ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
    enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
    for i, ezy in enumerate(ezy_list):
        seq_len = ezy.shape[0]
        ezy_padded[i, :seq_len, :] = ezy
        enzyme_mask[i, :, :seq_len] = 1

    # 底物特征tensor
    sbt_tensor = th.stack(sbt_list, dim=0)

    return ezy_padded, sbt_tensor, enzyme_mask


# ------------------------- 预测核心函数（按fold加载模型） -------------------------
def infer_kcat_km_by_fold(
    full_df, ezy_feats_list, sbt_feats, folds, invalid_idx,
    kcat_model_dir, km_model_dir, trained_model_dir,
    device, batch_size=8, alpha=0.5
):
    """按fold加载对应训练好的kcat-km模型，批量预测融合/左路/右路log(kcat/Km)值"""
    # 初始化预测结果数组（默认NaN）
    pred_fusion = np.full(len(full_df), np.nan, dtype=np.float32)
    pred_left = np.full(len(full_df), np.nan, dtype=np.float32)
    pred_right = np.full(len(full_df), np.nan, dtype=np.float32)

    # 按fold分组处理
    unique_folds = np.unique(folds)
    for fold in unique_folds:
        print(f"\n=== 处理Fold {fold} ===")
        # 1. 筛选当前fold的样本索引
        fold_mask = folds == fold
        fold_idx = np.where(fold_mask)[0].tolist()
        if not fold_idx:
            print(f"Fold {fold} 无样本，跳过")
            continue

        # 2. 筛选当前fold的特征
        fold_ezy_feats = [ezy_feats_list[i] for i in fold_idx]
        fold_sbt_feats = sbt_feats[fold_mask]

        # 3. 加载当前fold的训练好的kcat-km模型
        model_ckpt_path = os.path.join(trained_model_dir, f"fold{fold}_best_model.pth")
        if not os.path.exists(model_ckpt_path):
            print(f"警告：Fold {fold} 的模型文件不存在({model_ckpt_path})，该fold样本预测值为NaN")
            continue

        # 3.1 加载kcat预训练模型（右路）
        kcat_ckpt_path = os.path.join(kcat_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(kcat_ckpt_path):
            kcat_ckpt_path = os.path.join(kcat_model_dir, f"fold{fold}_best_model.pth")
        if not os.path.exists(kcat_ckpt_path):
            print(f"警告：Fold {fold} 的kcat预训练模型不存在，跳过该fold")
            continue
        kcat_ckpt = th.load(kcat_ckpt_path, map_location=device)

        # 3.2 初始化kcat模型的占位符子模型
        dummy_kcat_km_submodel = Model_Regression().to(device)
        dummy_km_submodel = Model_Regression().to(device)

        # 3.3 加载kcat模型压缩器权重
        if "kcat_km_compress_state" in kcat_ckpt and "km_compress_state" in kcat_ckpt:
            kcat_km_compress_state = kcat_ckpt["kcat_km_compress_state"]
            km_compress_state_kcat = kcat_ckpt["km_compress_state"]
        else:
            # 从km模型加载压缩器权重
            km_ckpt_temp = th.load(os.path.join(km_model_dir, f"fold{fold}_best_params.pth"), map_location=device)
            kcat_km_compress_state = km_ckpt_temp["compress_state"]
            km_compress_state_kcat = km_ckpt_temp["compress_state"]

        # 3.4 初始化kcat融合模型
        kcat_model = KcatOriginalActivityModel(
            kcat_km_model=dummy_kcat_km_submodel,
            Km_model=dummy_km_submodel,
            kcat_km_compress_state=kcat_km_compress_state,
            km_compress_state=km_compress_state_kcat,
            alpha=0.5,
            device=device
        ).to(device)
        kcat_model.load_state_dict(kcat_ckpt["model_state"])

        # 3.5 加载km预训练模型（右路）
        km_ckpt_path = os.path.join(km_model_dir, f"fold{fold}_best_params.pth")
        km_ckpt = th.load(km_ckpt_path, map_location=device)
        km_model = Model_Regression().to(device)
        km_model.load_state_dict(km_ckpt["model_state"])
        km_compress_state = km_ckpt["compress_state"]

        # 3.6 初始化kcat-km融合模型（预测模型）
        model = ActivityModel(
            kcat_model=kcat_model,
            km_model=km_model,
            km_compress_state=km_compress_state,
            alpha=alpha,
            device=device
        ).to(device)

        # 3.7 加载训练好的kcat-km模型权重
        trained_ckpt = th.load(model_ckpt_path, map_location=device)
        model.load_state_dict(trained_ckpt["model_state"])
        model.eval()  # 设为评估模式

        # 4. 创建预测数据集和dataloader
        infer_dataset = Mydatasets_Infer(fold_ezy_feats, fold_sbt_feats)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
            collate_fn=collate_fn_infer
        )

        # 5. 批量预测
        fold_fusion_pred = []
        fold_left_pred = []
        fold_right_pred = []
        with th.no_grad():
            for batch in infer_loader:
                ezy_padded, sbt_tensor, enzyme_mask = [x.to(device) for x in batch]
                # 模型输出：融合log(kcat/Km)、左路log(kcat/Km)、右路log(kcat/Km)
                fusion_pred, left_pred, right_pred = model(ezy_padded, sbt_tensor, enzyme_mask)
                # 维度调整
                fusion_pred = fusion_pred.squeeze(-1).cpu().numpy()
                left_pred = left_pred.squeeze(-1).cpu().numpy()
                right_pred = right_pred.squeeze(-1).cpu().numpy()
                # 收集结果
                fold_fusion_pred.extend(fusion_pred.tolist())
                fold_left_pred.extend(left_pred.tolist())
                fold_right_pred.extend(right_pred.tolist())

        # 6. 将预测结果赋值到对应位置
        pred_fusion[fold_idx] = fold_fusion_pred
        pred_left[fold_idx] = fold_left_pred
        pred_right[fold_idx] = fold_right_pred
        print(f"Fold {fold} 预测完成，样本数：{len(fold_idx)}")

    # 7. 无效样本保持NaN
    pred_fusion[invalid_idx] = np.nan
    pred_left[invalid_idx] = np.nan
    pred_right[invalid_idx] = np.nan

    return pred_fusion, pred_left, pred_right


# ------------------------- 主函数 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kcat-Km Model Inference (Fusion/Left/Right)",
                                     formatter_class=RawDescriptionHelpFormatter)
    # 数据路径
    parser.add_argument("-infer_fpath", type=str,
                        default="/mnt/usb3/wmx/kcat_km/case/kcat_km_final_valid_samples.pkl",
                        help="Path to kcat-km inference dataset (pkl)")
    # 模型路径（需和训练时一致）
    parser.add_argument("-kcat_model_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/kcat_km/model_kcat_2",
                        help="Pre-trained kcat model directory")
    parser.add_argument("-km_model_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/kcat_km/model_km_2",
                        help="Pre-trained km model directory")
    parser.add_argument("-trained_model_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/kcat_km/kcat_km/1_3",
                        help="Trained kcat-km model directory (fold*_best_model.pth)")
    # 超参数（需和训练时一致）
    parser.add_argument("-batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("-alpha", type=float, default=0.5, help="Left model weight (same as training)")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
    # 输出路径
    parser.add_argument("-out_fpath", type=str,
                        default="/mnt/usb3/wmx/kcat_km/case/predict.pkl",
                        help="Output path with prediction")
    args = parser.parse_args()

    # 1. 加载预测数据
    full_df, ezy_feats_list, sbt_feats, folds, invalid_idx = load_kcat_km_infer_data(args.infer_fpath)

    # 2. 按fold预测kcat-km值（融合/左路/右路）
    pred_fusion, pred_left, pred_right = infer_kcat_km_by_fold(
        full_df=full_df,
        ezy_feats_list=ezy_feats_list,
        sbt_feats=sbt_feats,
        folds=folds,
        invalid_idx=invalid_idx,
        kcat_model_dir=args.kcat_model_dir,
        km_model_dir=args.km_model_dir,
        trained_model_dir=args.trained_model_dir,
        device=args.device,
        batch_size=args.batch_size,
        alpha=args.alpha
    )

    # 3. 新增预测列到原始数据集
    full_df["pred_fusion_log10[kcat/Km]"] = pred_fusion
    full_df["pred_left_log10[kcat/Km]"] = pred_left
    full_df["pred_right_log10[kcat/Km]"] = pred_right
    print(f"\n预测完成！新增3列预测值：")
    print(f"  - pred_fusion_log10[kcat/Km]：融合模型预测值")
    print(f"  - pred_left_log10[kcat/Km]：左路模型预测值")
    print(f"  - pred_right_log10[kcat/Km]：右路模型预测值")

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
    print(f"融合模型有效预测数：{np.count_nonzero(~np.isnan(pred_fusion))}")
    print(f"左路模型有效预测数：{np.count_nonzero(~np.isnan(pred_left))}")
    print(f"右路模型有效预测数：{np.count_nonzero(~np.isnan(pred_right))}")
    print(f"无效预测数（NaN）：{np.count_nonzero(np.isnan(pred_fusion))}")
    print(f"预测成功率：{np.count_nonzero(~np.isnan(pred_fusion)) / len(full_df) * 100:.2f}%")