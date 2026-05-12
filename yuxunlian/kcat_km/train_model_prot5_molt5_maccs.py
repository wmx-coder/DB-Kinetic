import pandas as pd
import numpy as np
import torch as th
import copy
from torch.utils.data import DataLoader, Dataset
from model_prot5_molt5_maccs import Model_Regression, ActivityModel
from util_model2 import (
    EarlyStopping, write_logfile, out_results, evaluate, RMSELoss, rmse_loss
)
from argparse import RawDescriptionHelpFormatter
import argparse
import os
import datetime


# ------------------------- 数据集类 -------------------------
class Mydatasets(Dataset):
    def __init__(self, ezy_feats, sbt_feats, labels):
        self.ezy_feats = ezy_feats  # list of (seq_len, 1024)
        self.sbt_feats = sbt_feats  # (n_samples, 935)
        self.labels = labels        # (n_samples,)

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
        data_df = pd.read_pickle(fpath)[
            ["new_ezy_feat", "sbt_feat", "log10_kcat_over_Km", "fold"]
        ].copy()
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

        # 标签和折索引
        self.labels = data_df["log10_kcat_over_Km"].values
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

        train_loader = DataLoader(
            Mydatasets(train_ezy, train_sbt, train_label),
            batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True,
            collate_fn=self._collate_fn
        )
        valid_loader = DataLoader(
            Mydatasets(valid_ezy, valid_sbt, valid_label),
            batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=self._collate_fn
        )
        return train_loader, valid_idx, valid_loader

    def _collate_fn(self, batch):
        ezy_list, sbt_list, label_list = zip(*batch)
        batch_size = len(ezy_list)
        max_seq_len = max(ezy.shape[0] for ezy in ezy_list)

        ezy_padded = th.zeros((batch_size, max_seq_len, 1024), dtype=th.float32)
        enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
        for i, ezy in enumerate(ezy_list):
            seq_len = ezy.shape[0]
            ezy_padded[i, :seq_len, :] = ezy
            enzyme_mask[i, :, :seq_len] = 1

        sbt_tensor = th.stack(sbt_list, dim=0)
        label_tensor = th.stack(label_list, dim=0)

        return ezy_padded, sbt_tensor, label_tensor, enzyme_mask


# ------------------------- 训练/验证函数 -------------------------
def custom_train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_label, y_pred = [], []

    for batch in data_loader:
        ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
        _, _, pred_activity = model(ezy_feats, sbt_feats, enzyme_mask)
        pred_activity = pred_activity.squeeze(-1)

        loss = rmse_loss(pred_activity, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.cuda.empty_cache()

        with th.no_grad():
            total_loss += loss.item()
            y_label.append(labels.cpu().numpy())
            y_pred.append(pred_activity.cpu().numpy())

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    pcc, scc, r2, rmse = evaluate(y_label, y_pred)
    avg_loss = total_loss / len(data_loader)
    return np.array([pcc, scc, r2, rmse, avg_loss])


def custom_eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    y_label, y_pred = [], []

    with th.no_grad():
        for batch in data_loader:
            ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
            _, _, pred_activity = model(ezy_feats, sbt_feats, enzyme_mask)
            pred_activity = pred_activity.squeeze(-1)

            loss = rmse_loss(pred_activity, labels)
            total_loss += loss.item()
            y_label.append(labels.cpu().numpy())
            y_pred.append(pred_activity.cpu().numpy())

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    pcc, scc, r2, rmse = evaluate(y_label, y_pred)
    avg_loss = total_loss / len(data_loader)
    return y_pred, y_label, np.array([pcc, scc, r2, rmse, avg_loss])


# ------------------------- 主训练逻辑 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActivityModel Training with Path Config",
                                     formatter_class=RawDescriptionHelpFormatter)
    # 数据路径
    parser.add_argument("-data_fpath", type=str,
                        default="/mnt/usb3/code/wm/data/kcat_km/kcat_km_with_log_feats.pkl",
                        help="Dataset path")
    # 预训练模型路径
    parser.add_argument("-kcat_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_final_2",
                        help="Kcat pre-trained model directory")
    parser.add_argument("-km_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/km/km_model2_1",
                        help="Km pre-trained model directory")
    # 输出路径（日志、结果、模型）
    parser.add_argument("-log_dir", type=str,
                        default="logfile_model2_1",  # 日志目录（可相对/绝对路径）
                        help="Directory to save training logs")
    parser.add_argument("-results_dir", type=str,
                        default="results_model2_1",  # 结果目录
                        help="Directory to save evaluation results")
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_km/model2_1",  # 模型保存目录（你指定的路径）
                        help="Directory to save best ActivityModel")
    # 训练超参数
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-epochs", type=int, default=150, help="Max epochs")
    parser.add_argument("-alpha", type=float, default=0.5, help="Activity fusion weight")
    parser.add_argument("-dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
    # 早停参数
    parser.add_argument("-patience", type=int, default=20, help="EarlyStopping patience")
    parser.add_argument("-min_delta", type=float, default=0.001, help="EarlyStopping min delta")
    args = parser.parse_args()

    # 创建输出目录（确保不存在时自动创建）
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    # 加载数据集
    print(f"加载数据集：{args.data_fpath}")
    cv_data = CVDatasets(fpath=args.data_fpath, batch_size=args.batch_size)
    print(f"数据集加载完成，有效样本数：{len(cv_data.labels)}")

    # 10折交叉验证
    all_preds = []
    all_labels = []
    all_sample_indices = []

    for fold in range(10):
        print(f"\n" + "="*70)
        print(f"[Fold {fold}/{9}] 开始训练")
        print("="*70)

        # 加载预训练kcat模型
        kcat_model = Model_Regression().to(args.device)
        kcat_ckpt_path = os.path.join(args.kcat_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(kcat_ckpt_path):
            raise FileNotFoundError(f"Kcat模型不存在：{kcat_ckpt_path}")
        kcat_ckpt = th.load(kcat_ckpt_path, map_location=args.device)
        kcat_model.load_state_dict(kcat_ckpt["model_state"])

        # 加载预训练Km模型
        km_model = Model_Regression().to(args.device)
        km_ckpt_path = os.path.join(args.km_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(km_ckpt_path):
            raise FileNotFoundError(f"Km模型不存在：{km_ckpt_path}")
        km_ckpt = th.load(km_ckpt_path, map_location=args.device)
        km_model.load_state_dict(km_ckpt["model_state"])

        # 固定子模型参数（如需微调，注释这部分）
        for param in kcat_model.parameters():
            param.requires_grad = False
        for param in km_model.parameters():
            param.requires_grad = False

        # 初始化ActivityModel
        model = ActivityModel(
            kcat_model=kcat_model,
            km_model=km_model,
            rate=args.dropout_rate,
            alpha=args.alpha,
            device=args.device
        ).to(args.device)

        # 优化器
        optimizer = th.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        # 早停
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        best_model_state = None
        # start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_records = []

        # 单折训练循环
        for epoch in range(args.epochs):
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")
            # 获取加载器
            train_loader, valid_idx, valid_loader = cv_data.get_dataloader(fold)
            # 训练
            train_metrics = custom_train_epoch(model, train_loader, optimizer, args.device)
            # 验证
            valid_pred, valid_label, valid_metrics = custom_eval_epoch(model, valid_loader, args.device)

            # 记录日志
            log_entry = np.concatenate([
                np.array([epoch]),
                train_metrics,
                valid_metrics,
                np.array([1 if early_stopping.is_bestmodel else 0]),
                np.array([early_stopping.best_epoch])
            ])
            log_records.append(log_entry)
            # 写入日志文件（按折保存）
            write_logfile(
                epoch=epoch,
                record_data=log_records,
                logfile=os.path.join(args.log_dir, f"fold{fold}_log.csv")
            )

            # 早停检查与最优模型保存
            is_best, need_stop = early_stopping.check(epoch, valid_metrics[-1])
            if is_best:
                best_model_state = copy.deepcopy(model.state_dict())
                # 保存最优模型到指定目录
                best_model_path = os.path.join(args.model_save_dir, f"fold{fold}_best_model.pth")
                th.save({
                    "model_state": best_model_state,
                    "epoch": epoch,
                    "valid_loss": valid_metrics[-1]
                }, best_model_path)
                print(f"[Fold {fold}] 保存最优模型至：{best_model_path}")

            if need_stop:
                print(f"[Fold {fold}] 早停触发，终止于Epoch {epoch}")
                break

        # 记录训练时间
        # end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # with open(os.path.join(args.log_dir, f"fold{fold}_time.txt"), "w") as f:
        #     f.write(f"开始时间: {start_time}\n结束时间: {end_time}")

        # 加载最优模型并重新验证
        model.load_state_dict(best_model_state)
        final_valid_pred, final_valid_label, final_valid_metrics = custom_eval_epoch(model, valid_loader, args.device)
        print(f"[Fold {fold}] 最优模型验证结果：")
        print(f"  PCC: {final_valid_metrics[0]:.4f}, SCC: {final_valid_metrics[1]:.4f}")
        print(f"  R2: {final_valid_metrics[2]:.4f}, RMSE: {final_valid_metrics[3]:.4f}")

        # 保存单折结果到results_dir
        out_results(
            final_valid_metrics,
            os.path.join(args.results_dir, f"fold{fold}_results.csv")
        )

        # 收集全局结果
        all_preds.append(final_valid_pred)
        all_labels.append(final_valid_label)
        all_sample_indices.extend(valid_idx)

        print(f"[Fold {fold}] 训练完成\n" + "-"*70)

    # 汇总全局结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # # 保存最终预测-标签文件
    # final_pred_df = pd.DataFrame(
    #     data=np.column_stack([all_preds, all_labels]),
    #     index=all_sample_indices,
    #     columns=["pred_log10_kcat_over_Km", "label_log10_kcat_over_Km"]
    # )
    # final_pred_df.to_csv(os.path.join(args.results_dir, "final_pred_label.csv"), float_format="%.4f")

    # 计算并保存全局指标
    global_pcc, global_scc, global_r2, global_rmse = evaluate(all_labels, all_preds)
    global_metrics_df = pd.DataFrame(
        data=[[global_pcc, global_scc, global_r2, global_rmse]],
        columns=["PCC", "SCC", "R2", "RMSE"]
    )
    global_metrics_df.to_csv(os.path.join(args.results_dir, "final_results_model2_1.csv"), index=False, float_format="%.4f")

    print(f"\n" + "="*70)
    print("全局最终结果：")
    print(f"PCC: {global_pcc:.4f} | SCC: {global_scc:.4f} | R2: {global_r2:.4f} | RMSE: {global_rmse:.4f}")
    print("="*70)
    print(f"日志保存至：{args.log_dir}")
    print(f"结果保存至：{args.results_dir}")
    print(f"模型保存至：{args.model_save_dir}")