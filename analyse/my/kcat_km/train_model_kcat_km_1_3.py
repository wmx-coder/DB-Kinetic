import pandas as pd
import numpy as np
import torch as th
import copy
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
import os
import datetime

# 导入工具函数
from util_check import (
    EarlyStopping, out_results, evaluate, RMSELoss, rmse_loss
)

# 导入模型
from model_kcat_km_1 import Model_Regression, KcatOriginalActivityModel, ActivityModel_Freeze as ActivityModel


# ------------------------- 数据集类（无需修改，适配kcat/Km数据格式） -------------------------
class Mydatasets(Dataset):
    def __init__(self, ezy_feats, sbt_feats, labels):
        self.ezy_feats = ezy_feats  # list of (seq_len, 1024)
        self.sbt_feats = sbt_feats  # (n_samples, 935)
        self.labels = labels  # (n_samples,)  目标：log(kcat/Km)

    def __getitem__(self, idx):
        ezy = th.from_numpy(self.ezy_feats[idx]).float()
        sbt = th.from_numpy(self.sbt_feats[idx]).float()
        label = th.tensor(self.labels[idx]).float()
        return ezy, sbt, label

    def __len__(self):
        return len(self.labels)


class CVDatasets():
    def __init__(self, fpath, batch_size=32):
        self.batch_size = batch_size
        # 核心：加载kcat/Km数据集（标签为 log(kcat/Km)，已预定义fold列）
        data_df = pd.read_pickle(fpath)[["new_ezy_feat", "sbt_feat", "log10_kcat_over_Km", "fold"]].copy()
        self.data_index = data_df.index.tolist()

        # 酶特征校验
        self.ezy_feats_list = []
        for feat in data_df["new_ezy_feat"]:
            if not (isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024):
                raise ValueError(f"new_ezy_feat格式错误！需为(seq_len,1024)，当前{feat.shape}")
            self.ezy_feats_list.append(feat)

        # 底物特征校验
        self.sbt_feats = np.array([
            f for f in data_df["sbt_feat"]
            if isinstance(f, np.ndarray) and f.ndim == 1 and len(f) == 935
        ])
        if len(self.sbt_feats) != len(data_df):
            raise ValueError("部分sbt_feat维度不是935，请检查数据！")

        # 标签处理（log(kcat/Km)）
        self.labels = data_df["log10_kcat_over_Km"].values
        if np.isnan(self.labels).any():
            raise ValueError("标签中存在NaN值，请处理后再训练")
        self.folds = data_df["fold"].values.astype(int)

        # 拆分10折索引（基于预定义fold列，无随机）
        self.split_index_dict = {}
        for fold in range(10):
            valid_mask = self.folds == fold
            self.split_index_dict[fold] = [
                np.where(~valid_mask)[0].tolist(),
                np.where(valid_mask)[0].tolist()
            ]

    def get_dataloader(self, fold):
        train_idx, valid_idx = self.split_index_dict[fold]
        print(f"\n[Fold {fold}] 训练样本数: {len(train_idx)}, 验证样本数: {len(valid_idx)}")

        train_ezy = [self.ezy_feats_list[i] for i in train_idx]
        train_sbt = self.sbt_feats[train_idx]
        train_label = self.labels[train_idx]
        valid_ezy = [self.ezy_feats_list[i] for i in valid_idx]
        valid_sbt = self.sbt_feats[valid_idx]
        valid_label = self.labels[valid_idx]

        train_loader = DataLoader(
            Mydatasets(train_ezy, train_sbt, train_label),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            collate_fn=self._collate_fn,
        )
        valid_loader = DataLoader(
            Mydatasets(valid_ezy, valid_sbt, valid_label),
            batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,
            collate_fn=self._collate_fn,
        )
        return train_loader, valid_idx, valid_loader

    def _collate_fn(self, batch):
        ezy_list, sbt_list, label_list = zip(*batch)
        batch_size = len(ezy_list)
        max_seq_len = max(ezy.shape[0] for ezy in ezy_list)

        # 酶特征padding + 掩码
        ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
        enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
        for i, ezy in enumerate(ezy_list):
            seq_len = ezy.shape[0]
            ezy_padded[i, :seq_len, :] = ezy
            enzyme_mask[i, :, :seq_len] = 1

        # 底物特征 + 标签
        sbt_tensor = th.stack(sbt_list, dim=0)
        label_tensor = th.stack(label_list, dim=0)

        return ezy_padded, sbt_tensor, label_tensor, enzyme_mask


# ------------------------- 训练/验证函数（无修改，保持原功能） -------------------------
def custom_train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_label, y_fusion_pred, y_left_pred, y_right_pred = [], [], [], []

    for batch in data_loader:
        ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
        # 模型输出：final_log_ratio (log(kcat/Km)), pred_left_log_ratio, pred_right_log_ratio
        final_log_ratio, pred_left_log_ratio, pred_right_log_ratio = model(ezy_feats, sbt_feats, enzyme_mask)
        final_log_ratio = final_log_ratio.squeeze(-1)
        pred_left_log_ratio = pred_left_log_ratio.squeeze(-1)
        pred_right_log_ratio = pred_right_log_ratio.squeeze(-1)

        # 计算损失（目标：log(kcat/Km)）
        loss = rmse_loss(final_log_ratio, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.cuda.empty_cache()

        # 收集结果
        with th.no_grad():
            total_loss += loss.item()
            y_label.append(labels.cpu().numpy())
            y_fusion_pred.append(final_log_ratio.cpu().numpy())
            y_left_pred.append(pred_left_log_ratio.cpu().numpy())
            y_right_pred.append(pred_right_log_ratio.cpu().numpy())

    # 计算训练指标
    y_label = np.concatenate(y_label)
    y_fusion_pred = np.concatenate(y_fusion_pred)
    y_left_pred = np.concatenate(y_left_pred)
    y_right_pred = np.concatenate(y_right_pred)

    fusion_pcc, fusion_scc, fusion_r2, fusion_rmse = evaluate(y_label, y_fusion_pred)
    left_pcc, left_scc, left_r2, left_rmse = evaluate(y_label, y_left_pred)
    right_pcc, right_scc, right_r2, right_rmse = evaluate(y_label, y_right_pred)
    avg_loss = total_loss / len(data_loader)

    return np.array([
        fusion_pcc, fusion_scc, fusion_r2, fusion_rmse,  # 融合指标（0-3）
        left_pcc, left_scc, left_r2, left_rmse,          # 左路指标（4-7）
        right_pcc, right_scc, right_r2, right_rmse,      # 右路指标（8-11）
        avg_loss                                         # 训练损失（12）
    ])


def custom_eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    y_label, y_fusion_pred, y_left_pred, y_right_pred = [], [], [], []

    with th.no_grad():
        for batch in data_loader:
            ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
            final_log_ratio, pred_left_log_ratio, pred_right_log_ratio = model(ezy_feats, sbt_feats, enzyme_mask)
            final_log_ratio = final_log_ratio.squeeze(-1)
            pred_left_log_ratio = pred_left_log_ratio.squeeze(-1)
            pred_right_log_ratio = pred_right_log_ratio.squeeze(-1)

            loss = rmse_loss(final_log_ratio, labels)
            total_loss += loss.item()

            y_label.append(labels.cpu().numpy())
            y_fusion_pred.append(final_log_ratio.cpu().numpy())
            y_left_pred.append(pred_left_log_ratio.cpu().numpy())
            y_right_pred.append(pred_right_log_ratio.cpu().numpy())

    # 计算验证指标
    y_label = np.concatenate(y_label)
    y_fusion_pred = np.concatenate(y_fusion_pred)
    y_left_pred = np.concatenate(y_left_pred)
    y_right_pred = np.concatenate(y_right_pred)

    fusion_pcc, fusion_scc, fusion_r2, fusion_rmse = evaluate(y_label, y_fusion_pred)
    left_pcc, left_scc, left_r2, left_rmse = evaluate(y_label, y_left_pred)
    right_pcc, right_scc, right_r2, right_rmse = evaluate(y_label, y_right_pred)
    avg_loss = total_loss / len(data_loader)

    return (
        y_fusion_pred, y_left_pred, y_right_pred, y_label,
        np.array([
            fusion_pcc, fusion_scc, fusion_r2, fusion_rmse,  # 融合指标（0-3）
            left_pcc, left_scc, left_r2, left_rmse,          # 左路指标（4-7）
            right_pcc, right_scc, right_r2, right_rmse,      # 右路指标（8-11）
            avg_loss                                         # 验证损失（12）
        ])
    )


# ------------------------- 10折主训练逻辑（核心修改：早停日志顺序） -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10折训练 - 预测 log(kcat/Km)（左路+右路kcat-Km+融合）",
                                     formatter_class=RawDescriptionHelpFormatter)
    # 数据路径（修改为你的kcat/Km数据集路径）
    parser.add_argument("-data_fpath", type=str,
                        default="/mnt/usb3/code/wm/data/kcat_km/kcat_km_with_log_feats.pkl",
                        help="kcat/Km数据集路径（需含 log_kcat_km 和 fold 列）")
    # 预训练模型路径（修改为你的实际路径）
    parser.add_argument("-kcat_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_km/model_kcat_2",
                        help="logkcat 预训练模型目录（融合模型）")
    parser.add_argument("-km_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_km/model_km_2",
                        help="logKm 预训练模型目录（单模型）")
    # 输出路径（自定义，避免覆盖原有结果）
    parser.add_argument("-log_dir", type=str, default="logfile_kcat_km_1_3", help="日志目录")
    parser.add_argument("-results_dir", type=str, default="results_kcat_km_1_3", help="结果目录")
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_km/kcat_km/1_3",
                        help="融合模型保存目录")
    # 超参数（保持原配置，可根据需求调整）
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-5, help="学习率（冻结版专用）")
    parser.add_argument("-epochs", type=int, default=150, help="最大epoch")
    parser.add_argument("-alpha", type=float, default=0.5, help="左路权重（0-1）")
    parser.add_argument("-device", type=str, default="cuda:0", help="设备（cuda:0/cpu）")
    # 早停参数
    parser.add_argument("-patience", type=int, default=20, help="早停patience")
    parser.add_argument("-min_delta", type=float, default=0.001, help="早停最小变化量")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    # 加载数据集（kcat/Km 标签，基于预定义fold列）
    print(f"加载kcat/Km数据集：{args.data_fpath}")
    cv_data = CVDatasets(fpath=args.data_fpath, batch_size=args.batch_size)
    print(f"数据集加载完成，总样本数：{len(cv_data.labels)}")

    # 全局结果容器
    global_left_preds = []
    global_right_preds = []
    global_fusion_preds = []
    global_true_labels = []

    # 10折训练循环
    for fold in range(10):
        print(f"\n" + "=" * 80)
        print(f"[Fold {fold}/{9}] 开始训练")
        print("=" * 80)

        # ---------------------- 核心修改：加载预训练模型（右路：kcat模型 + km模型） ----------------------
        # 1. 加载 kcat模型（融合框架：KcatOriginalActivityModel）
        kcat_ckpt_path = os.path.join(args.kcat_model_dir, f"fold{fold}_best_model.pth")  # 融合模型权重名通常是best_model.pth
        if not os.path.exists(kcat_ckpt_path):
            # 若权重名是best_params.pth，替换下面一行（根据你的实际权重文件名调整）
            kcat_ckpt_path = os.path.join(args.kcat_model_dir, f"fold{fold}_best_params.pth")
            if not os.path.exists(kcat_ckpt_path):
                raise FileNotFoundError(f"kcat融合模型不存在：{kcat_ckpt_path}")
        kcat_ckpt = th.load(kcat_ckpt_path, map_location=args.device)

        # 初始化kcat模型的占位符子模型（和训练kcat时一致）
        dummy_kcat_km_submodel = Model_Regression().to(args.device)  # kcat模型内部的kcat_km子模型
        dummy_km_submodel = Model_Regression().to(args.device)       # kcat模型内部的km子模型

        # 加载kcat模型需要的压缩器权重（从kcat权重文件或km模型文件中获取）
        if "kcat_km_compress_state" in kcat_ckpt and "km_compress_state" in kcat_ckpt:
            # 情况1：kcat权重文件中已保存压缩器权重（推荐）
            kcat_km_compress_state = kcat_ckpt["kcat_km_compress_state"]
            km_compress_state_kcat = kcat_ckpt["km_compress_state"]
            print("✅ 从kcat权重文件加载内部压缩器权重")
        else:
            # 情况2：从km模型文件中加载（需确保km模型路径正确）
            km_ckpt_temp = th.load(os.path.join(args.km_model_dir, f"fold{fold}_best_params.pth"), map_location=args.device)
            kcat_km_compress_state = km_ckpt_temp["compress_state"]  # 复用km的压缩器权重（根据实际调整）
            km_compress_state_kcat = km_ckpt_temp["compress_state"]
            print("✅ 从km模型文件加载kcat内部压缩器权重")

        # 初始化kcat融合模型（必须和训练kcat时的参数一致！）
        kcat_model = KcatOriginalActivityModel(
            kcat_km_model=dummy_kcat_km_submodel,
            Km_model=dummy_km_submodel,
            kcat_km_compress_state=kcat_km_compress_state,
            km_compress_state=km_compress_state_kcat,
            alpha=0.5,  # 必须和训练kcat时的alpha一致！
            device=args.device
        )
        # 加载kcat模型预训练权重
        kcat_model.load_state_dict(kcat_ckpt["model_state"])
        print(f"✅ 成功加载第{fold}折kcat融合模型")

        # 2. 加载 km模型（单框架：Model_Regression）
        km_ckpt_path = os.path.join(args.km_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(km_ckpt_path):
            raise FileNotFoundError(f"km单模型不存在：{km_ckpt_path}")
        km_ckpt = th.load(km_ckpt_path, map_location=args.device)
        km_model = Model_Regression().to(args.device)
        km_model.load_state_dict(km_ckpt["model_state"])
        km_compress_state = km_ckpt["compress_state"]  # km模型需要的外部压缩器权重
        print(f"✅ 成功加载第{fold}折km单模型和压缩器")

        # 3. 初始化融合模型（预测 log(kcat/Km)：左路+右路(kcat - km)）
        model = ActivityModel(
            kcat_model=kcat_model,          # 右路：kcat融合模型（预测logkcat）
            km_model=km_model,              # 右路：km单模型（预测logKm）
            km_compress_state=km_compress_state,  # km模型的外部压缩器权重
            alpha=args.alpha,
            device=args.device
        ).to(args.device)

        # 4. 优化器（仅优化可训练参数：左路模型+左路压缩器）
        optimizer = th.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        # 5. 早停初始化
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        best_model_state = None
        log_records = []

        # 6. 获取当前fold的加载器
        train_loader, valid_idx, valid_loader = cv_data.get_dataloader(fold=fold)

        # 7. 训练循环（核心修改：早停检查 → 日志记录）
        for epoch in range(args.epochs):
            print(f"[Epoch {epoch + 1}/{args.epochs}]")

            # 训练
            train_metrics = custom_train_epoch(model, train_loader, optimizer, args.device)

            # 验证
            valid_fusion_pred, valid_left_pred, valid_right_pred, valid_label, valid_metrics = custom_eval_epoch(
                model, valid_loader, args.device
            )

            # ---------------------- 核心修复：先早停检查，再记录日志 ----------------------
            # 早停检查（更新is_bestmodel和best_epoch）
            print(f"[Epoch {epoch}] 早停调试：valid_loss={valid_metrics[12]:.6f} | 历史最佳损失={early_stopping.min_loss:.6f}")
            is_best, need_stop = early_stopping.check(epoch, valid_metrics[12])  # 按验证损失判断

            # 保存最优模型
            if is_best:
                best_model_state = copy.deepcopy(model.state_dict())
                best_model_path = os.path.join(args.model_save_dir, f"fold{fold}_best_model.pth")
                th.save({
                    "model_state": best_model_state,
                    "epoch": epoch,
                    "valid_loss": valid_metrics[12],
                    "valid_fusion_pcc": valid_metrics[0],
                    "valid_left_pcc": valid_metrics[4],
                    "valid_right_pcc": valid_metrics[8],
                    "alpha": args.alpha,
                    "lr": args.lr,
                }, best_model_path)
                print(f"[Epoch {epoch + 1}] 保存最优模型至：{best_model_path}")

            # 记录日志（使用更新后的is_bestmodel和best_epoch）
            log_entry = np.concatenate([
                np.array([epoch]),
                # 验证集指标（融合+左路+右路）
                np.array([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]]),  # 融合
                np.array([valid_metrics[4], valid_metrics[5], valid_metrics[6], valid_metrics[7]]),  # 左路
                np.array([valid_metrics[8], valid_metrics[9], valid_metrics[10], valid_metrics[11]]),# 右路
                # 训练集融合指标
                np.array([train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3]]),
                # 损失
                np.array([train_metrics[12], valid_metrics[12]]),
                # 早停相关（最新值）
                np.array([1 if early_stopping.is_bestmodel else 0]),
                np.array([early_stopping.best_epoch])
            ])
            log_records.append(log_entry)

            # 写入日志文件
            log_header = "epoch," \
                         "valid_fusion_pcc,valid_fusion_scc,valid_fusion_r2,valid_fusion_rmse," \
                         "valid_left_pcc,valid_left_scc,valid_left_r2,valid_left_rmse," \
                         "valid_right_pcc,valid_right_scc,valid_right_r2,valid_right_rmse," \
                         "train_fusion_pcc,train_fusion_scc,train_fusion_r2,train_fusion_rmse," \
                         "train_loss,valid_loss,is_best_model,current_best_epoch"
            if epoch == 0:
                with open(os.path.join(args.log_dir, f"fold{fold}_log.csv"), 'w') as f:
                    f.write(log_header + '\n')
            with open(os.path.join(args.log_dir, f"fold{fold}_log.csv"), 'a') as f:
                f.write(','.join(map(lambda x: f"{x:.6f}", log_entry)) + '\n')

            # 早停终止判断
            if need_stop:
                print(f"早停触发，终止于Epoch {epoch + 1}")
                break

        # 8. 加载最优模型重新验证
        model.load_state_dict(best_model_state)
        final_fusion_pred, final_left_pred, final_right_pred, final_label, final_metrics = custom_eval_epoch(
            model, valid_loader, args.device
        )

        # 9. 收集当前fold结果到全局容器
        global_fusion_preds.append(final_fusion_pred)
        global_left_preds.append(final_left_pred)
        global_right_preds.append(final_right_pred)
        global_true_labels.append(final_label)

        # 10. 保存当前fold最终结果
        final_results_df = pd.DataFrame({
            "model_type": ["融合模型", "左路模型", "右路模型"],
            "fold": [fold] * 3,
            "PCC": [final_metrics[0], final_metrics[4], final_metrics[8]],
            "SCC": [final_metrics[1], final_metrics[5], final_metrics[9]],
            "R2": [final_metrics[2], final_metrics[6], final_metrics[10]],
            "RMSE": [final_metrics[3], final_metrics[7], final_metrics[11]]
        })
        final_results_path = os.path.join(args.results_dir, f"fold{fold}_final_results.csv")
        final_results_df.to_csv(final_results_path, index=False, float_format="%.6f")

        # 11. 保存详细预测结果
        final_pred_df = pd.DataFrame({
            "fold": [fold] * len(valid_idx),
            "sample_idx": valid_idx,
            "label_log_kcat_km": final_label,
            "pred_fusion_log_kcat_km": final_fusion_pred,
            "pred_left_log_kcat_km": final_left_pred,
            "pred_right_log_kcat_km": final_right_pred,
            "fusion_residual": final_fusion_pred - final_label,
            "left_residual": final_left_pred - final_label,
            "right_residual": final_right_pred - final_label
        })
        final_pred_path = os.path.join(args.results_dir, f"fold{fold}_final_pred_detail.csv")
        final_pred_df.to_csv(final_pred_path, index=False, float_format="%.6f")

        # 12. 打印当前fold结果汇总
        print(f"\n[Fold {fold}] 最终验证结果汇总")
        print("-" * 70)
        print(f"{'模型类型':<12} {'PCC':<10} {'SCC':<10} {'R2':<10} {'RMSE':<10}")
        print("-" * 70)
        print(f"{'融合模型':<12} {final_metrics[0]:.4f}    {final_metrics[1]:.4f}    {final_metrics[2]:.4f}    {final_metrics[3]:.4f}")
        print(f"{'左路模型':<12} {final_metrics[4]:.4f}    {final_metrics[5]:.4f}    {final_metrics[6]:.4f}    {final_metrics[7]:.4f}")
        print(f"{'右路模型':<12} {final_metrics[8]:.4f}    {final_metrics[9]:.4f}    {final_metrics[10]:.4f}    {final_metrics[11]:.4f}")
        print("=" * 80)

    # ------------------------- 全局结果汇总 -------------------------
    print(f"\n" + "=" * 80)
    print("🎉 10折交叉训练全部完成！开始计算全局结果")
    print("=" * 80)

    # 拼接所有fold的结果
    global_fusion_preds = np.concatenate(global_fusion_preds)
    global_left_preds = np.concatenate(global_left_preds)
    global_right_preds = np.concatenate(global_right_preds)
    global_true_labels = np.concatenate(global_true_labels)

    # 计算三路全局指标
    left_pcc, left_scc, left_r2, left_rmse = evaluate(global_true_labels, global_left_preds)
    right_pcc, right_scc, right_r2, right_rmse = evaluate(global_true_labels, global_right_preds)
    fusion_pcc, fusion_scc, fusion_r2, fusion_rmse = evaluate(global_true_labels, global_fusion_preds)

    # 保存全局结果
    global_results = pd.DataFrame({
        "model_type": ["左路模型", "右路模型", "融合模型"],
        "PCC": [left_pcc, right_pcc, fusion_pcc],
        "SCC": [left_scc, right_scc, fusion_scc],
        "R2": [left_r2, right_r2, fusion_r2],
        "RMSE": [left_rmse, right_rmse, fusion_rmse]
    })
    global_results_path = os.path.join(args.results_dir, "final_global_results.csv")
    global_results.to_csv(global_results_path, index=False, float_format="%.4f")

    # 打印全局结果
    print(f"\n全局最终结果：")
    print("-" * 70)
    print(f"{'模型类型':<12} {'PCC':<10} {'SCC':<10} {'R2':<10} {'RMSE':<10}")
    print("-" * 70)
    print(f"{'左路模型':<12} {left_pcc:.4f}    {left_scc:.4f}    {left_r2:.4f}    {left_rmse:.4f}")
    print(f"{'右路模型':<12} {right_pcc:.4f}    {right_scc:.4f}    {right_r2:.4f}    {right_rmse:.4f}")
    print(f"{'融合模型':<12} {fusion_pcc:.4f}    {fusion_scc:.4f}    {fusion_r2:.4f}    {fusion_rmse:.4f}")
    print("=" * 80)
    print(f"全局结果文件保存至：{global_results_path}")

    # 输出核心文件路径汇总
    print(f"\n📁 所有结果文件汇总：")
    print(f"- 训练日志目录：{args.log_dir}")
    print(f"- 指标/预测结果目录：{args.results_dir}")
    print(f"- 最优模型目录：{args.model_save_dir}")
    print("=" * 80)