import torch as th
import pandas as pd
import numpy as np
import os
import copy
from argparse import RawDescriptionHelpFormatter
import argparse
from torch.utils.data import DataLoader, Dataset

# 导入自定义模块（需确保路径正确）
# 注意：运行此脚本前，请确保 model_freeze.py 和 util_check.py 在当前路径或 PYTHONPATH 中
from model_freeze import Model_Regression, ActivityModel_Freeze as ActivityModel
from util_check import RMSELoss

# 抑制冗余日志
import logging

logging.basicConfig(level=logging.ERROR)
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


# ------------------------- 数据集类（适配fold分组预测） -------------------------
class KcatInferDataset(Dataset):
    def __init__(self, ezy_feats, sbt_feats, folds):
        """
        初始化预测数据集
        :param ezy_feats: 酶特征列表，每个元素为(seq_len, 1024)的np.array
        :param sbt_feats: 底物特征数组，shape=(n_samples, 935)
        :param folds: 样本对应的fold值数组，shape=(n_samples,)
        """
        self.ezy_feats = ezy_feats
        self.sbt_feats = sbt_feats
        self.folds = folds

    def __getitem__(self, idx):
        ezy = th.from_numpy(self.ezy_feats[idx]).float()
        sbt = th.from_numpy(self.sbt_feats[idx]).float()
        fold = self.folds[idx]
        return ezy, sbt, fold

    def __len__(self):
        return len(self.folds)


def collate_fn(batch):
    """自定义collate_fn，处理酶特征的padding和掩码，与训练脚本保持一致"""
    ezy_list, sbt_list, fold_list = zip(*batch)
    batch_size = len(ezy_list)
    max_seq_len = max(ezy.shape[0] for ezy in ezy_list)

    # 酶特征padding + 掩码
    ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
    enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
    for i, ezy in enumerate(ezy_list):
        seq_len = ezy.shape[0]
        ezy_padded[i, :seq_len, :] = ezy
        enzyme_mask[i, :, :seq_len] = 1

    # 底物特征 + fold值
    sbt_tensor = th.stack(sbt_list, dim=0)
    fold_tensor = th.tensor(fold_list, dtype=th.int64)

    return ezy_padded, sbt_tensor, enzyme_mask, fold_tensor


# ------------------------- 模型加载与预测函数 -------------------------
def load_model(fold, model_save_dir, kcat_km_model_dir, km_model_dir, alpha, device):
    """
    加载指定fold的模型
    """
    # 1. 加载kcat/km和Km的预训练模型
    kcat_km_ckpt_path = os.path.join(kcat_km_model_dir, f"fold{fold}_best_params.pth")
    if not os.path.exists(kcat_km_ckpt_path):
        raise FileNotFoundError(f"kcat/km模型文件不存在：{kcat_km_ckpt_path}")
    kcat_km_ckpt = th.load(kcat_km_ckpt_path, map_location=device)
    kcat_km_model = Model_Regression().to(device)
    kcat_km_model.load_state_dict(kcat_km_ckpt["model_state"])
    kcat_km_compress_state = kcat_km_ckpt["compress_state"]

    km_ckpt_path = os.path.join(km_model_dir, f"fold{fold}_best_params.pth")
    if not os.path.exists(km_ckpt_path):
        raise FileNotFoundError(f"Km模型文件不存在：{km_ckpt_path}")
    km_ckpt = th.load(km_ckpt_path, map_location=device)
    km_model = Model_Regression().to(device)
    km_model.load_state_dict(km_ckpt["model_state"])
    km_compress_state = km_ckpt["compress_state"]

    # 2. 初始化融合模型
    model = ActivityModel(
        kcat_km_model=kcat_km_model,
        Km_model=km_model,
        kcat_km_compress_state=kcat_km_compress_state,
        km_compress_state=km_compress_state,
        alpha=alpha,
        device=device
    ).to(device)

    # 3. 加载当前fold的kcat最优模型权重
    model_ckpt_path = os.path.join(model_save_dir, f"fold{fold}_best_model.pth")
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"kcat模型权重文件不存在：{model_ckpt_path}")
    model_ckpt = th.load(model_ckpt_path, map_location=device)
    model.load_state_dict(model_ckpt["model_state"])

    # 4. 设置模型为评估模式
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def infer_by_fold(data_df, model_save_dir, kcat_km_model_dir, km_model_dir, alpha, batch_size, device):
    """
    按fold分组进行预测
    """
    # 1. 数据预处理与校验
    # 酶特征校验
    ezy_feats_list = []
    for feat in data_df["new_ezy_feat"]:
        if not (isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024):
            raise ValueError(f"new_ezy_feat格式错误！需为(seq_len,1024)，当前{feat.shape}")
        ezy_feats_list.append(feat)

    # 底物特征校验
    sbt_feats = np.array([
        f for f in data_df["sbt_feat"]
        if isinstance(f, np.ndarray) and f.ndim == 1 and len(f) == 935
    ])
    if len(sbt_feats) != len(data_df):
        raise ValueError("部分sbt_feat维度不是935，请检查数据！")

    # fold值校验
    folds = data_df["fold"].values.astype(int)
    if not np.all((folds >= 0) & (folds <= 9)):
        raise ValueError("fold值必须在0-9范围内！")

    # 2. 创建数据集和数据加载器
    dataset = KcatInferDataset(ezy_feats_list, sbt_feats, folds)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 3. 按fold分组预测（避免重复加载模型）
    pred_logkcat = np.zeros(len(data_df))  # 存储最终预测结果
    model_cache = {}  # 缓存已加载的模型，key=fold，value=model

    with th.no_grad():
        for batch_idx, (ezy_padded, sbt_tensor, enzyme_mask, fold_tensor) in enumerate(dataloader):
            # 批次内按fold分组
            batch_folds = fold_tensor.unique().tolist()
            for fold in batch_folds:
                # 筛选当前fold的样本索引
                fold_mask = (fold_tensor == fold)
                if not fold_mask.any():
                    continue
                batch_idx_fold = th.where(fold_mask)[0].cpu().numpy()

                # 加载模型（优先从缓存读取）
                if fold not in model_cache:
                    model_cache[fold] = load_model(
                        fold=fold,
                        model_save_dir=model_save_dir,
                        kcat_km_model_dir=kcat_km_model_dir,
                        km_model_dir=km_model_dir,
                        alpha=alpha,
                        device=device
                    )
                model = model_cache[fold]

                # 提取当前fold的样本数据
                ezy_batch = ezy_padded[fold_mask].to(device)
                sbt_batch = sbt_tensor[fold_mask].to(device)
                mask_batch = enzyme_mask[fold_mask].to(device)

                # 预测
                final_logkcat, _, _ = model(ezy_batch, sbt_batch, mask_batch)
                final_logkcat = final_logkcat.squeeze(-1).cpu().numpy()

                # 计算全局索引并赋值
                start_idx = batch_idx * batch_size
                global_idx = start_idx + batch_idx_fold
                pred_logkcat[global_idx] = final_logkcat

    return pred_logkcat


# ------------------------- 主函数 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按样本fold值匹配对应模型进行kcat预测（非10折平均）",
                                     formatter_class=RawDescriptionHelpFormatter)
    # 数据路径
    parser.add_argument("-inp_fpath", type=str,
                        default="/mnt/usb3/wmx/kcat/kcat-data_feats_complete.pkl",
                        help="待预测数据集路径（需包含fold列、new_ezy_feat、sbt_feat）")
    # 模型路径
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/kcat/models_model_freeze",
                        help="训练好的kcat模型保存目录（含fold0-fold9的best_model.pth）")
    parser.add_argument("-kcat_km_model_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/kcat_km/model3_1",
                        help="kcat/km预训练模型目录")
    parser.add_argument("-km_model_dir", type=str,
                        default="/mnt/usb1/wmx/catapro/models/km/km_model2_1",
                        help="Km预训练模型目录")
    # 输出路径
    parser.add_argument("-out_fpath", type=str,
                        default="kcat.pkl",
                        help="预测结果输出文件路径")
    # 超参数
    parser.add_argument("-alpha", type=float, default=0.5, help="融合模型左路权重（与训练一致）")
    parser.add_argument("-batch_size", type=int, default=8, help="预测批次大小（与训练一致）")
    parser.add_argument("-device", type=str, default="cuda:0", help="运行设备（cuda:0/cpu）")

    args = parser.parse_args()

    # 1. 加载待预测数据
    print(f"加载待预测数据：{args.inp_fpath}")
    if args.inp_fpath.endswith(".pkl"):
        data_df = pd.read_pickle(args.inp_fpath)
    elif args.inp_fpath.endswith(".csv"):
        data_df = pd.read_csv(args.inp_fpath)
        # 若sbt_feat/new_ezy_feat是字符串格式，需转换为np.array（示例）
        # import ast
        # data_df["sbt_feat"] = data_df["sbt_feat"].apply(lambda x: np.array(ast.literal_eval(x)))
        # data_df["new_ezy_feat"] = data_df["new_ezy_feat"].apply(lambda x: np.array(ast.literal_eval(x)))
    else:
        raise ValueError("仅支持.pkl或.csv格式的输入文件！")

    # 校验必要列
    required_cols = ["fold", "new_ezy_feat", "sbt_feat"]
    if not all(col in data_df.columns for col in required_cols):
        raise ValueError(f"输入数据必须包含列：{required_cols}")

    # 2. 按fold匹配模型进行预测
    print("开始按fold分组预测...")
    pred_logkcat = infer_by_fold(
        data_df=data_df,
        model_save_dir=args.model_save_dir,
        kcat_km_model_dir=args.kcat_km_model_dir,
        km_model_dir=args.km_model_dir,
        alpha=args.alpha,
        batch_size=args.batch_size,
        device=args.device
    )

    # 3. 整理预测结果
    print("整理预测结果...")
    result_df = data_df.copy()
    result_df["pred_log10_kcat"] = pred_logkcat

    # --- [修改点] 移除不需要的特征列 ---
    cols_to_drop = []
    target_cols = ["ezy_feat", "new_ezy_feat", "sbt_feat"]

    for col in target_cols:
        if col in result_df.columns:
            cols_to_drop.append(col)

    if cols_to_drop:
        result_df = result_df.drop(columns=cols_to_drop)
        print(f"已从结果中移除列：{cols_to_drop}")
    else:
        print("未找到需要移除的特征列。")
    # ----------------------------------

    # 可选：还原为原始kcat值（10^pred_log10_kcat）
    # result_df["pred_kcat"] = 10 ** result_df["pred_log10_kcat"]

    # 4. 保存结果
    result_df.to_pickle(args.out_fpath)
    print(f"预测结果已保存至：{args.out_fpath}")

    # 5. 打印预测结果统计信息
    print("\n预测结果统计：")
    print(f"- 总样本数：{len(result_df)}")
    # 注意：如果fold列也被误删了（上面逻辑不会删fold），这里会报错，但上面只删了特征列
    if 'fold' in result_df.columns:
        print(f"- 各fold样本数：\n{result_df['fold'].value_counts().sort_index()}")

    if 'pred_log10_kcat' in result_df.columns:
        print(f"- 预测log10(kcat)均值：{result_df['pred_log10_kcat'].mean():.4f}")
        print(f"- 预测log10(kcat)标准差：{result_df['pred_log10_kcat'].std():.4f}")