import pandas as pd
import numpy as np
import torch as th
import copy
import os
import datetime
from torch.utils.data import DataLoader, Dataset
from model_molt_maccs import Model_Regression  # 模型文件无需修改（已适配ESM维度）
from util_model2 import (
    EarlyStopping,
    write_logfile, out_results, evaluate, RMSELoss
)
from argparse import RawDescriptionHelpFormatter
import argparse


# ------------------------- 数据集类（核心修改：适配ESM索引+按需加载.pt） -------------------------
class Mydatasets(Dataset):
    def __init__(self, esm_ids, sbt_feats, labels, esm_feat_dir, target_layer=33):
        self.esm_ids = esm_ids  # ESM特征文件ID（数字索引）
        self.sbt_feats = sbt_feats
        self.labels = labels
        self.esm_feat_dir = esm_feat_dir  # ESM .pt文件目录
        self.target_layer = target_layer  # ESM特征层

    def __getitem__(self, idx):
        # 1. 通过ID拼接ESM .pt文件路径
        esm_id = self.esm_ids[idx]
        esm_pt_path = os.path.join(self.esm_feat_dir, f"unique_kcat_km_{esm_id}.pt")

        # 2. 加载ESM特征（按需加载）
        try:
            esm_data = th.load(esm_pt_path, map_location="cpu")
            esm_feat = esm_data["representations"][self.target_layer][1:-1, :].float()
        except Exception as e:
            raise ValueError(f"加载ESM文件失败 (ID={esm_id}, Path={esm_pt_path}): {str(e)}")

        # 3. 底物特征
        sbt = th.from_numpy(self.sbt_feats[idx]).float()

        # 4. 标签
        label = th.tensor(self.labels[idx]).float()

        return esm_feat, sbt, label

    def __len__(self):
        return len(self.labels)


class CVDatasets():
    def __init__(self, fpath, esm_feat_dir, batch_size=32):
        self.batch_size = batch_size
        self.esm_feat_dir = esm_feat_dir  # ESM .pt文件目录
        # 读取带esm_id的数据集（需提前生成）
        data_df = pd.read_pickle(fpath)[["esm_id", "sbt_feat", "log10_kcat_over_Km", "fold"]].copy()
        self.data_index = data_df.index.tolist()

        # 验证esm_id有效性
        invalid_esm_id = data_df["esm_id"].isna() | (data_df["esm_id"] < 0)
        if invalid_esm_id.any():
            raise ValueError(f"存在无效ESM ID，数量：{invalid_esm_id.sum()}")
        self.esm_ids = data_df["esm_id"].values.astype(int)

        # 底物特征验证
        valid_sbt = []
        for f in data_df["sbt_feat"]:
            if isinstance(f, np.ndarray) and f.ndim == 1 and len(f) == 935:
                valid_sbt.append(f)
            else:
                raise ValueError(f"morgan_feat格式错误，需为(2048,)的np.ndarray")
        self.sbt_feats = np.array(valid_sbt)

        self.labels = data_df["log10_kcat_over_Km"].values
        self.folds = data_df["fold"].values.astype(int)

        # 折划分
        self.split_index_dict = {}
        for fold in range(10):
            valid_mask = self.folds == fold
            self.split_index_dict[fold] = [
                np.where(~valid_mask)[0].tolist(),
                np.where(valid_mask)[0].tolist()
            ]

    def _get_subset(self, indices):
        esm_ids_subset = self.esm_ids[indices]
        sbt_subset = self.sbt_feats[indices]
        label_subset = self.labels[indices]
        return esm_ids_subset, sbt_subset, label_subset

    def _collate_fn(self, batch):
        esm_list, sbt_list, label_list = zip(*batch)
        batch_size = len(esm_list)
        max_seq_len = max(esm.shape[0] for esm in esm_list)

        # ESM特征padding + 掩码（适配1280维）
        esm_padded = th.zeros((batch_size, max_seq_len, 1280), dtype=th.float32)
        enzyme_mask = th.zeros((batch_size, 1, max_seq_len), dtype=th.float32)
        for i, esm in enumerate(esm_list):
            seq_len = esm.shape[0]
            esm_padded[i, :seq_len, :] = esm
            enzyme_mask[i, :, :seq_len] = 1

        sbt_tensor = th.stack(sbt_list, dim=0)
        label_tensor = th.stack(label_list, dim=0)

        return esm_padded, sbt_tensor, label_tensor, enzyme_mask

    def get_dataloader(self, fold):
        train_idx, valid_idx = self.split_index_dict[fold]
        print(f"\n[get_dataloader] Fold{fold} - 训练样本数: {len(train_idx)}, 验证样本数: {len(valid_idx)}")

        train_data = self._get_subset(train_idx)
        valid_data = self._get_subset(valid_idx)

        # 构建Dataset（传入ESM目录）
        train_dataset = Mydatasets(
            *train_data,
            esm_feat_dir=self.esm_feat_dir,
            target_layer=33
        )
        valid_dataset = Mydatasets(
            *valid_data,
            esm_feat_dir=self.esm_feat_dir,
            target_layer=33
        )

        # 构建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True,
            collate_fn=self._collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=self._collate_fn
        )
        return train_loader, valid_idx, valid_loader


# ------------------------- 训练/验证函数（无修改） -------------------------
def custom_train_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    y_label, y_pred = [], []
    rmse_loss = RMSELoss()

    for batch in data_loader:
        ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
        pred, _ = model(sbt_feats, ezy_feats, enzyme_mask=enzyme_mask)
        pred = pred.squeeze(-1)

        loss = rmse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.cuda.empty_cache()

        total_loss += loss.item()
        y_label.append(labels.cpu().detach().numpy())
        y_pred.append(pred.cpu().detach().numpy())

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    pcc, scc, r2, rmse = evaluate(y_label, y_pred)
    avg_loss = total_loss / len(data_loader)
    return np.array([pcc, scc, r2, rmse, avg_loss])


def custom_eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    y_label, y_pred = [], []
    rmse_loss = RMSELoss()

    with th.no_grad():
        for batch in data_loader:
            ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
            pred, _ = model(sbt_feats, ezy_feats, enzyme_mask=enzyme_mask)
            pred = pred.squeeze(-1)

            loss = rmse_loss(pred, labels)
            total_loss += loss.item()
            y_label.append(labels.cpu().detach().numpy())
            y_pred.append(pred.cpu().detach().numpy())

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    pcc, scc, r2, rmse = evaluate(y_label, y_pred)
    avg_loss = total_loss / len(data_loader)
    return y_pred, y_label, np.array([pcc, scc, r2, rmse, avg_loss])


# ------------------------- 主训练逻辑（仅新增ESM目录参数） -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kcat_km Training (ESM Feature)",
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-data_fpath", type=str,
                        default="/mnt/usb3/code/wm/data/kcat_km/kcat_km_v1_with_fold_esm_id.pkl",
                        help="带ESM ID的数据集路径")
    parser.add_argument("-esm_feat_dir", type=str,
                        default="/mnt/usb3/code/wm/esm/kcat_km/",
                        help="ESM .pt文件存储目录")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("-epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/yuxunlian_models/kcat/kcat_km/model_molt_maccs_2",
                        help="Directory to save best models")
    parser.add_argument("-log_dir", type=str, default="logfile_model_molt_maccs_2", help="Log directory")
    parser.add_argument("-results_dir", type=str, default="results_model_molt_maccs_2", help="Results directory")
    parser.add_argument("-patience", type=int, default=20, help="EarlyStopping patience")
    parser.add_argument("-min_delta", type=float, default=0.001, help="EarlyStopping min_delta")
    args = parser.parse_args()

    # 创建目录
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 初始化数据集（传入ESM目录）
    cv_data = CVDatasets(
        fpath=args.data_fpath,
        esm_feat_dir=args.esm_feat_dir,
        batch_size=args.batch_size
    )

    all_pred_label = []
    all_indices = []

    # 10折交叉验证
    for fold in range(10):
        print(f"\n=== Start Training Fold {fold} ===")
        train_loader, valid_idx, valid_loader = cv_data.get_dataloader(fold)

        # 初始化模型
        model = Model_Regression().to(args.device)
        optimizer = th.optim.Adam(
            model.parameters(),
            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5
        )

        # 早停
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        best_model_state = copy.deepcopy(model.state_dict())

        # 日志头
        log_header = [
            "epoch", "train_pcc", "train_scc", "train_r2", "train_rmse", "train_loss",
            "valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_loss",
            "is_best_model", "current_best_epoch"
        ]
        record_data = []

        # 训练循环
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            # 训练
            train_metrics = custom_train_epoch(
                model=model, data_loader=train_loader, optimizer=optimizer,
                device=args.device, epoch=epoch
            )
            # 验证
            valid_pred, valid_label, valid_metrics = custom_eval_epoch(
                model=model, data_loader=valid_loader, device=args.device
            )

            # 早停检查
            is_best, need_stop = early_stopping.check(epoch, valid_metrics[-1])
            if is_best:
                best_model_state = copy.deepcopy(model.state_dict())
                # 保存最优模型
                model_save_path = os.path.join(args.model_save_dir, f"fold{fold}_best_params.pth")
                th.save({
                    "model_state": best_model_state,
                    "best_epoch": epoch,
                    "valid_rmse": valid_metrics[3],
                    "valid_loss": valid_metrics[4]
                }, model_save_path)

            # 记录日志
            current_best_epoch = early_stopping.best_epoch
            log_entry = np.concatenate([
                np.array([epoch]),
                train_metrics,
                valid_metrics,
                np.array([1 if is_best else 0]),
                np.array([current_best_epoch])
            ])
            record_data.append(log_entry)

            # 写入日志
            write_logfile(
                epoch=epoch,
                record_data=record_data,
                logfile=os.path.join(args.log_dir, f"logfile_fold{fold}.csv")
            )

            # 早停终止
            if need_stop:
                print(f"Fold {fold} - Early Stopping Triggered! Stop at Epoch {epoch}")
                break

        # 加载最优模型
        model.load_state_dict(best_model_state)
        # 最终验证
        final_valid_pred, final_valid_label, final_valid_metrics = custom_eval_epoch(
            model=model, data_loader=valid_loader, device=args.device
        )

        # 打印结果
        best_epoch = early_stopping.best_epoch
        best_result_log = f"\n=== Fold {fold} 最优模型最终结果 ==="
        best_result_log += f"\n1. 最优Epoch：{best_epoch}"
        best_result_log += f"\n2. 验证集指标："
        best_result_log += f"\n   - PCC: {final_valid_metrics[0]:.4f}"
        best_result_log += f"\n   - SCC: {final_valid_metrics[1]:.4f}"
        best_result_log += f"\n   - R²: {final_valid_metrics[2]:.4f}"
        best_result_log += f"\n   - RMSE: {final_valid_metrics[3]:.4f}"
        best_result_log += f"\n   - 平均损失: {final_valid_metrics[4]:.4f}"
        print(best_result_log)

        # 保存结果
        with open(os.path.join(args.log_dir, f"logfile_fold{fold}.csv"), 'a') as f:
            f.write(f"\n# {best_result_log.replace(chr(10), chr(10) + '# ')}")
        out_results(final_valid_metrics, os.path.join(args.results_dir, f"results_fold{fold}.csv"))

        # 汇总结果
        all_pred_label.append(
            np.concatenate([final_valid_pred.reshape(-1, 1), final_valid_label.reshape(-1, 1)], axis=1)
        )
        all_indices.extend(valid_idx)
        print(f"=== Finish Training Fold {fold} ===")

    # 全局结果汇总
    all_pred_label = np.concatenate(all_pred_label, axis=0)
    final_pred_df = pd.DataFrame(
        all_pred_label, index=all_indices, columns=["pred_log10_kcat_km", "label_log10_kcat_km"]
    )

    final_pcc, final_scc, final_r2, final_rmse = evaluate(all_pred_label[:, 1], all_pred_label[:, 0])
    final_metrics = pd.DataFrame(
        np.array([[final_pcc, final_scc, final_r2, final_rmse]]),
        columns=["PCC", "SCC", "R2", "RMSE"]
    )
    final_metrics.to_csv(os.path.join(args.results_dir, "final_results_model_molt_maccs_2.csv"), index=False,
                         float_format="%.4f")
    print(f"\n全局最终结果：PCC={final_pcc:.4f}, SCC={final_scc:.4f}, R2={final_r2:.4f}, RMSE={final_rmse:.4f}")