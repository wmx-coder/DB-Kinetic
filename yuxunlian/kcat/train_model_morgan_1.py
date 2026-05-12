import pandas as pd
import numpy as np
import torch as th
import copy
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
import os
import datetime
from util_check import (
    EarlyStopping, out_results, evaluate, RMSELoss, rmse_loss
)

# 导入修改后的Kcat模型
from model_morgan import Model_Regression, ActivityModel_Freeze as ActivityModel


# ------------------------- 数据集类（适配768维molt5_feat） -------------------------
class Mydatasets(Dataset):
    def __init__(self, ezy_feats, sbt_feats, labels):
        self.ezy_feats = ezy_feats  # list of (seq_len, 1024)
        self.sbt_feats = sbt_feats  # (n_samples, 768) 适配molt5_feat
        self.labels = labels  # (n_samples,)  log10(kcat)

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
        # 🔴 加载molt5_feat（768维）而非sbt_feat（935维）
        data_df = pd.read_pickle(fpath)[["new_ezy_feat", "morgan_feat", "log10_kcat", "fold"]].copy()
        self.data_index = data_df.index.tolist()

        # 酶特征校验
        self.ezy_feats_list = []
        for feat in data_df["new_ezy_feat"]:
            if not (isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024):
                raise ValueError(f"new_ezy_feat格式错误！需为(seq_len,1024)，当前{feat.shape}")
            self.ezy_feats_list.append(feat)

        # 底物特征校验（768维molt5_feat）
        self.sbt_feats = np.array([
            f for f in data_df["morgan_feat"]
            if isinstance(f, np.ndarray) and f.ndim == 1 and len(f) == 2048
        ])
        if len(self.sbt_feats) != len(data_df):
            raise ValueError("部分molt5_feat维度不是768，请检查数据！")

        # 标签处理
        self.labels = data_df["log10_kcat"].values
        if np.isnan(self.labels).any():
            raise ValueError("标签中存在NaN值，请处理后再训练")
        self.folds = data_df["fold"].values.astype(int)

        # 拆分10折索引
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

        # 🔴 降低num_workers缓解资源压力
        train_loader = DataLoader(
            Mydatasets(train_ezy, train_sbt, train_label),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            collate_fn=self._collate_fn
        )
        valid_loader = DataLoader(
            Mydatasets(valid_ezy, valid_sbt, valid_label),
            batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,
            collate_fn=self._collate_fn
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


# ------------------------- 训练/验证函数 -------------------------
def custom_train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_label, y_fusion_pred, y_left_pred, y_right_pred = [], [], [], []

    for batch in data_loader:
        ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
        final_logkcat, pred_left_logkcat, pred_right_logkcat = model(ezy_feats, sbt_feats, enzyme_mask)
        final_logkcat = final_logkcat.squeeze(-1)
        pred_left_logkcat = pred_left_logkcat.squeeze(-1)
        pred_right_logkcat = pred_right_logkcat.squeeze(-1)

        # 计算损失并反向传播
        loss = rmse_loss(final_logkcat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.cuda.empty_cache()

        # 收集结果
        with th.no_grad():
            total_loss += loss.item()
            y_label.append(labels.cpu().numpy())
            y_fusion_pred.append(final_logkcat.cpu().numpy())
            y_left_pred.append(pred_left_logkcat.cpu().numpy())
            y_right_pred.append(pred_right_logkcat.cpu().numpy())

    # 计算指标
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
        left_pcc, left_scc, left_r2, left_rmse,  # 左路指标（4-7）
        right_pcc, right_scc, right_r2, right_rmse,  # 右路指标（8-11）
        avg_loss  # 平均损失（12）
    ])


def custom_eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    y_label, y_fusion_pred, y_left_pred, y_right_pred = [], [], [], []

    with th.no_grad():
        for batch in data_loader:
            ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
            final_logkcat, pred_left_logkcat, pred_right_logkcat = model(ezy_feats, sbt_feats, enzyme_mask)
            final_logkcat = final_logkcat.squeeze(-1)
            pred_left_logkcat = pred_left_logkcat.squeeze(-1)
            pred_right_logkcat = pred_right_logkcat.squeeze(-1)

            loss = rmse_loss(final_logkcat, labels)
            total_loss += loss.item()

            y_label.append(labels.cpu().numpy())
            y_fusion_pred.append(final_logkcat.cpu().numpy())
            y_left_pred.append(pred_left_logkcat.cpu().numpy())
            y_right_pred.append(pred_right_logkcat.cpu().numpy())

    # 计算指标
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
            fusion_pcc, fusion_scc, fusion_r2, fusion_rmse,
            left_pcc, left_scc, left_r2, left_rmse,
            right_pcc, right_scc, right_r2, right_rmse,
            avg_loss
        ])
    )


# ------------------------- 10折主训练逻辑 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kcat融合模型训练（内置底物压缩层）",
                                     formatter_class=RawDescriptionHelpFormatter)
    # 数据路径
    parser.add_argument("-data_fpath", type=str,
                        default="/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete_with_morgan.pkl",
                        help="kcat数据集路径（含molt5_feat）")
    # 预训练模型路径
    parser.add_argument("-kcat_km_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/yuxunlian_models/kcat/kcat_km/model_morgan_1",
                        help="kcat/km预训练模型目录")
    parser.add_argument("-km_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/yuxunlian_models/kcat/km/model_morgan_1",
                        help="Km预训练模型目录")
    # 输出路径
    parser.add_argument("-log_dir", type=str, default="logfile_model_morgan_1", help="日志目录")
    parser.add_argument("-results_dir", type=str, default="results_model_morgan_1", help="结果目录")
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/yuxunlian_models/kcat/kcat/model_morgan_1",
                        help="模型保存目录")
    # 超参数
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("-epochs", type=int, default=150, help="最大epoch")
    parser.add_argument("-alpha", type=float, default=0.5, help="左路权重")
    parser.add_argument("-device", type=str, default="cuda:0", help="设备")
    # 早停参数
    parser.add_argument("-patience", type=int, default=20, help="早停patience")
    parser.add_argument("-min_delta", type=float, default=0.001, help="早停最小变化量")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    # 加载数据集
    print(f"加载kcat数据集：{args.data_fpath}")
    cv_data = CVDatasets(fpath=args.data_fpath, batch_size=args.batch_size)
    print(f"数据集加载完成，总样本数：{len(cv_data.labels)}")

    # 10折训练循环
    for fold in range(10):
        print(f"\n" + "=" * 80)
        print(f"[Fold {fold}/{9}] 开始训练")
        print("=" * 80)

        # 1. 加载预训练Kcat/Km模型（无需单独加载压缩器）
        kcat_km_ckpt_path = os.path.join(args.kcat_km_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(kcat_km_ckpt_path):
            raise FileNotFoundError(f"kcat/km模型不存在：{kcat_km_ckpt_path}")
        kcat_km_ckpt = th.load(kcat_km_ckpt_path, map_location=args.device)
        kcat_km_model = Model_Regression().to(args.device)
        kcat_km_model.load_state_dict(kcat_km_ckpt["model_state"])

        km_ckpt_path = os.path.join(args.km_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(km_ckpt_path):
            raise FileNotFoundError(f"Km模型不存在：{km_ckpt_path}")
        km_ckpt = th.load(km_ckpt_path, map_location=args.device)
        km_model = Model_Regression().to(args.device)
        km_model.load_state_dict(km_ckpt["model_state"])

        # 2. 初始化融合模型（内置压缩层，无需外部压缩器）
        model = ActivityModel(
            kcat_km_model=kcat_km_model,
            Km_model=km_model,
            alpha=args.alpha,
            device=args.device
        ).to(args.device)

        # 3. 优化器（仅训练左路参数）
        optimizer = th.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        # 4. 早停初始化
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        best_model_state = copy.deepcopy(model.state_dict())  # 修正：兜底初始化
        log_records = []

        # 5. 获取数据加载器
        train_loader, valid_idx, valid_loader = cv_data.get_dataloader(fold=fold)

        # 6. 训练循环
        for epoch in range(args.epochs):
            print(f"[Epoch {epoch + 1}/{args.epochs}]")

            # 训练
            train_metrics = custom_train_epoch(model, train_loader, optimizer, args.device)

            # 验证
            valid_fusion_pred, valid_left_pred, valid_right_pred, valid_label, valid_metrics = custom_eval_epoch(
                model, valid_loader, args.device
            )

            # 早停检查（关键：先检查早停，再记录日志）
            is_best, need_stop = early_stopping.check(epoch, valid_metrics[12])

            # 新增打印：核心早停判定逻辑
            print(
                f"[早停判定] Epoch{epoch + 1} | 验证损失：{valid_metrics[12]:.6f} | 当前最优损失：{early_stopping.min_loss:.6f} | "
                f"阈值（最优-0.001）：{early_stopping.min_loss - args.min_delta:.6f} | is_best：{is_best} | 当前最优Epoch：{early_stopping.best_epoch}")

            # 记录日志（修正：早停检查后再记录，保证is_best和best_epoch是最新的）
            log_entry = np.concatenate([
                np.array([epoch]),
                valid_metrics[0:4],  # 融合指标
                valid_metrics[4:8],  # 左路指标
                valid_metrics[8:12],  # 右路指标
                train_metrics[0:4],  # 训练融合指标
                train_metrics[12:13],  # 训练损失
                valid_metrics[12:13],  # 验证损失
                np.array([1 if is_best else 0]),  # 用最新的is_best
                np.array([early_stopping.best_epoch])  # 用最新的best_epoch
            ])
            log_records.append(log_entry)

            # 写入日志
            log_header = "epoch," \
                         "valid_fusion_pcc,valid_fusion_scc,valid_fusion_r2,valid_fusion_rmse," \
                         "valid_left_pcc,valid_left_scc,valid_left_r2,valid_left_rmse," \
                         "valid_right_pcc,valid_right_scc,valid_right_r2,valid_right_rmse," \
                         "train_fusion_pcc,train_fusion_scc,train_fusion_r2,train_fusion_rmse," \
                         "train_loss,valid_loss,is_best_model,current_best_epoch"
            if epoch == 0:
                with open(os.path.join(args.log_dir, f"fold{fold}_log.csv"), 'w') as f:
                    f.write(log_header + '\n')
            # 新增打印：日志记录值（验证是否和判定值一致）
            print(f"[日志记录] is_best_model：{log_entry[-2]} | current_best_epoch：{log_entry[-1]}")

            # 修正日志格式化：区分整数和浮点数
            formatted_log = []
            for x in log_entry:
                if x in [log_entry[0], log_entry[-2], log_entry[-1]]:  # epoch、is_best、best_epoch
                    formatted_log.append(f"{int(x)}")
                else:
                    formatted_log.append(f"{x:.6f}")
            with open(os.path.join(args.log_dir, f"fold{fold}_log.csv"), 'a') as f:
                f.write(','.join(formatted_log) + '\n')

            # 保存最优模型
            if is_best:
                best_model_state = copy.deepcopy(model.state_dict())
                best_model_path = os.path.join(args.model_save_dir, f"fold{fold}_best_model.pth")
                th.save({
                    "model_state": best_model_state,
                    "epoch": epoch + 1,  # 修正：保存1开始的epoch
                    "valid_loss": valid_metrics[12],
                    "valid_fusion_pcc": valid_metrics[0],
                    "alpha": args.alpha,
                    "lr": args.lr
                }, best_model_path)
                print(f"[Epoch {epoch + 1}] 保存最优模型至：{best_model_path} | 最优损失更新为：{valid_metrics[12]:.6f}")

            if need_stop:
                print(f"早停触发，终止于Epoch {epoch + 1}")
                break

        # 7. 加载最优模型验证（新增容错：避免best_model_state为None）
        if best_model_state is None:
            print(f"警告：Fold {fold} 未找到最优模型，使用最后一轮模型")
            best_model_state = model.state_dict()
        model.load_state_dict(best_model_state)
        final_fusion_pred, final_left_pred, final_right_pred, final_label, final_metrics = custom_eval_epoch(
            model, valid_loader, args.device
        )

        # 8. 解析最终指标
        final_fusion_pcc = final_metrics[0]
        final_fusion_scc = final_metrics[1]
        final_fusion_r2 = final_metrics[2]
        final_fusion_rmse = final_metrics[3]
        final_left_pcc = final_metrics[4]
        final_left_scc = final_metrics[5]
        final_left_r2 = final_metrics[6]
        final_left_rmse = final_metrics[7]
        final_right_pcc = final_metrics[8]
        final_right_scc = final_metrics[9]
        final_right_r2 = final_metrics[10]
        final_right_rmse = final_metrics[11]

        # 9. 保存结果
        final_results_df = pd.DataFrame({
            "model_type": ["融合模型", "左路模型", "右路模型"],
            "fold": [fold] * 3,
            "PCC": [final_fusion_pcc, final_left_pcc, final_right_pcc],
            "SCC": [final_fusion_scc, final_left_scc, final_right_scc],
            "R2": [final_fusion_r2, final_left_r2, final_right_r2],
            "RMSE": [final_fusion_rmse, final_left_rmse, final_right_rmse]
        })
        final_results_path = os.path.join(args.results_dir, f"fold{fold}_final_results.csv")
        final_results_df.to_csv(final_results_path, index=False, float_format="%.6f")

        # 10. 保存详细预测结果
        final_pred_df = pd.DataFrame({
            "fold": [fold] * len(valid_idx),
            "sample_idx": valid_idx,
            "label_log10_kcat": final_label,
            "pred_fusion_log10_kcat": final_fusion_pred,
            "pred_left_log10_kcat": final_left_pred,
            "pred_right_log10_kcat": final_right_pred,
            "fusion_residual": final_fusion_pred - final_label,
            "left_residual": final_left_pred - final_label,
            "right_residual": final_right_pred - final_label
        })
        # final_pred_path = os.path.join(args.results_dir, f"fold{fold}_final_pred_detail.csv")
        # final_pred_df.to_csv(final_pred_path, index=False, float_format="%.6f")

        # 11. 打印结果
        print(f"\n[Fold {fold}] 最终验证结果汇总")
        print("-" * 70)
        print(f"{'模型类型':<12} {'PCC':<10} {'SCC':<10} {'R2':<10} {'RMSE':<10}")
        print("-" * 70)
        print(
            f"{'融合模型':<12} {final_fusion_pcc:.4f}    {final_fusion_scc:.4f}    {final_fusion_r2:.4f}    {final_fusion_rmse:.4f}")
        print(
            f"{'左路模型':<12} {final_left_pcc:.4f}    {final_left_scc:.4f}    {final_left_r2:.4f}    {final_left_rmse:.4f}")
        print(
            f"{'右路模型':<12} {final_right_pcc:.4f}    {final_right_scc:.4f}    {final_right_r2:.4f}    {final_right_rmse:.4f}")
        print("=" * 80)

        # 新增：每折结束清理GPU缓存
        th.cuda.empty_cache()
        del model, kcat_km_model, km_model

    # ====================== 新增：全局结果汇总 ======================
    print(f"\n" + "=" * 80)
    print("📊 开始统计10折全局结果...")
    print("=" * 80)

    # 1. 收集所有fold的最终指标
    all_fusion_metrics = []  # 融合模型：PCC/SCC/R2/RMSE
    all_left_metrics = []  # 左路模型：PCC/SCC/R2/RMSE
    all_right_metrics = []  # 右路模型：PCC/SCC/R2/RMSE

    for fold in range(10):
        # 读取当前fold的最终结果
        fold_result_path = os.path.join(args.results_dir, f"fold{fold}_final_results.csv")
        fold_df = pd.read_csv(fold_result_path)

        # 提取融合/左路/右路指标
        fusion_row = fold_df[fold_df["model_type"] == "融合模型"].iloc[0]
        left_row = fold_df[fold_df["model_type"] == "左路模型"].iloc[0]
        right_row = fold_df[fold_df["model_type"] == "右路模型"].iloc[0]

        all_fusion_metrics.append([fusion_row["PCC"], fusion_row["SCC"], fusion_row["R2"], fusion_row["RMSE"]])
        all_left_metrics.append([left_row["PCC"], left_row["SCC"], left_row["R2"], left_row["RMSE"]])
        all_right_metrics.append([right_row["PCC"], right_row["SCC"], right_row["R2"], right_row["RMSE"]])

    # 2. 转换为numpy数组并计算均值+标准差
    all_fusion_metrics = np.array(all_fusion_metrics)
    all_left_metrics = np.array(all_left_metrics)
    all_right_metrics = np.array(all_right_metrics)

    # 计算均值和标准差（保留4位小数）
    fusion_mean = np.round(np.mean(all_fusion_metrics, axis=0), 4)
    fusion_std = np.round(np.std(all_fusion_metrics, axis=0), 4)
    left_mean = np.round(np.mean(all_left_metrics, axis=0), 4)
    left_std = np.round(np.std(all_left_metrics, axis=0), 4)
    right_mean = np.round(np.mean(all_right_metrics, axis=0), 4)
    right_std = np.round(np.std(all_right_metrics, axis=0), 4)

    # 3. 构建全局汇总DataFrame
    global_summary_df = pd.DataFrame({
        "model_type": ["融合模型", "左路模型", "右路模型"],
        "PCC_mean": [fusion_mean[0], left_mean[0], right_mean[0]],
        "PCC_std": [fusion_std[0], left_std[0], right_std[0]],
        "SCC_mean": [fusion_mean[1], left_mean[1], right_mean[1]],
        "SCC_std": [fusion_std[1], left_std[1], right_std[1]],
        "R2_mean": [fusion_mean[2], left_mean[2], right_mean[2]],
        "R2_std": [fusion_std[2], left_std[2], right_std[2]],
        "RMSE_mean": [fusion_mean[3], left_mean[3], right_mean[3]],
        "RMSE_std": [fusion_std[3], left_std[3], right_std[3]]
    })

    # 4. 保存全局汇总结果
    global_summary_path = os.path.join(args.results_dir, "global_10fold_summary.csv")
    global_summary_df.to_csv(global_summary_path, index=False, float_format="%.4f")

    # 5. 打印全局汇总结果
    print(f"\n🚀 10折全局结果汇总（均值±标准差）")
    print("-" * 80)
    print(f"{'模型类型':<12} {'PCC':<15} {'SCC':<15} {'R2':<15} {'RMSE':<15}")
    print("-" * 80)
    print(
        f"{'融合模型':<12} {f'{fusion_mean[0]}±{fusion_std[0]}':<15} {f'{fusion_mean[1]}±{fusion_std[1]}':<15} {f'{fusion_mean[2]}±{fusion_std[2]}':<15} {f'{fusion_mean[3]}±{fusion_std[3]}':<15}")
    print(
        f"{'左路模型':<12} {f'{left_mean[0]}±{left_std[0]}':<15} {f'{left_mean[1]}±{left_std[1]}':<15} {f'{left_mean[2]}±{left_std[2]}':<15} {f'{left_mean[3]}±{left_std[3]}':<15}")
    print(
        f"{'右路模型':<12} {f'{right_mean[0]}±{right_std[0]}':<15} {f'{right_mean[1]}±{right_std[1]}':<15} {f'{right_mean[2]}±{right_std[2]}':<15} {f'{right_mean[3]}±{right_std[3]}':<15}")
    print("-" * 80)
    print(f"📁 全局汇总文件已保存至：{global_summary_path}")

    # ====================== 训练完成提示 ======================
    print(f"\n🎉 10折交叉训练全部完成！")
    print(f"📁 所有结果文件汇总：")
    print(f"- 训练日志目录：{args.log_dir}（10个fold分别的训练日志）")
    print(f"- 指标/预测结果目录：{args.results_dir}（含每个fold的最终指标、详细预测、全局汇总）")
    print(f"- 最优模型目录：{args.model_save_dir}（10个fold的最优模型权重）")
    print(f"\n💡 后续操作建议：")
    print(f"1. 查看全局汇总文件 {global_summary_path}，分析模型整体性能")
    print(f"2. 对比纯冻结版与部分微调版的10折均值，选择最优模型架构")
    print(f"3. 基于详细预测结果，分析模型误差分布（如残差较大的样本特征）")
    print("=" * 80)