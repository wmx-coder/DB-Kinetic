import pandas as pd
import numpy as np
import torch as th
import copy  # 新增：用于深拷贝
from torch.utils.data import DataLoader, Dataset
from model2_2 import Model_Regression
from util_model2 import (
    EarlyStopping,
    write_logfile, out_results, evaluate, RMSELoss
)
from argparse import RawDescriptionHelpFormatter
import argparse
import os
import datetime


# ------------------------- 数据集类（不变） -------------------------
class Mydatasets(Dataset):
    def __init__(self, ezy_feats, sbt_feats, labels):
        self.ezy_feats = ezy_feats
        self.sbt_feats = sbt_feats
        self.labels = labels

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
        data_df = pd.read_pickle(fpath)[["new_ezy_feat", "sbt_feat", "log10_kcat", "fold"]].copy()
        self.data_index = data_df.index.tolist()

        self.ezy_feats_list = []
        for feat in data_df["new_ezy_feat"]:
            if not (isinstance(feat, np.ndarray) and feat.ndim == 2 and feat.shape[1] == 1024):
                raise ValueError(f"new_ezy_feat 格式错误！需为 (seq_len,1024)，当前 {feat.shape}")
            self.ezy_feats_list.append(feat)

        self.sbt_feats = np.array([
            f for f in data_df["sbt_feat"]
            if isinstance(f, np.ndarray) and f.ndim == 1 and len(f) == 935
        ])

        self.labels = data_df["log10_kcat"].values
        self.folds = data_df["fold"].values.astype(int)

        self.split_index_dict = {}
        for fold in range(10):
            valid_mask = self.folds == fold
            self.split_index_dict[fold] = [
                np.where(~valid_mask)[0].tolist(),
                np.where(valid_mask)[0].tolist()
            ]

    def get_dataloader(self, fold):
        train_idx, valid_idx = self.split_index_dict[fold]
        print(f"\n[get_dataloader] Fold{fold} - 训练样本数: {len(train_idx)}, 验证样本数: {len(valid_idx)}")

        train_data = self._get_subset(train_idx)
        valid_data = self._get_subset(valid_idx)

        train_loader = DataLoader(
            Mydatasets(*train_data),
            batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True,
            collate_fn=self._collate_fn
        )
        valid_loader = DataLoader(
            Mydatasets(*valid_data),
            batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=self._collate_fn
        )
        return train_loader, valid_idx, valid_loader

    def _get_subset(self, indices):
        ezy_subset = [self.ezy_feats_list[i] for i in indices]
        sbt_subset = self.sbt_feats[indices]
        label_subset = self.labels[indices]
        return ezy_subset, sbt_subset, label_subset

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


# ------------------------- 训练/验证函数（新增训练后参数打印） -------------------------
def custom_train_epoch(model, data_loader, optimizer, device, sbt_compress, epoch):
    model.train()
    total_loss = 0.0
    y_label, y_pred = [], []
    rmse_loss = RMSELoss()

    for batch in data_loader:
        ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
        reactions = sbt_compress(sbt_feats).unsqueeze(1)
        pred, _ = model(reactions, ezy_feats, enzyme_mask=enzyme_mask)
        pred = pred.squeeze(-1)

        loss = rmse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.cuda.empty_cache()

        total_loss += loss.item()
        y_label.append(labels.cpu().detach().numpy())
        y_pred.append(pred.cpu().detach().numpy())

    # 新增：打印训练后的参数哈希（验证参数是否更新）
    train_model_hash = hash(str(model.state_dict()["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
    # print(f"Epoch {epoch} 训练后模型参数哈希：{train_model_hash}")

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    pcc, scc, r2, rmse = evaluate(y_label, y_pred)
    avg_loss = total_loss / len(data_loader)
    return np.array([pcc, scc, r2, rmse, avg_loss])


def custom_eval_epoch(model, data_loader, device, sbt_compress):
    model.eval()
    total_loss = 0.0
    y_label, y_pred = [], []
    rmse_loss = RMSELoss()

    with th.no_grad():
        for batch in data_loader:
            ezy_feats, sbt_feats, labels, enzyme_mask = [x.to(device) for x in batch]
            reactions = sbt_compress(sbt_feats).unsqueeze(1)
            pred, _ = model(reactions, ezy_feats, enzyme_mask=enzyme_mask)
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


# ------------------------- 主训练逻辑（核心：强制深拷贝） -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kcat Training (With Deepcopy)",
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-data_fpath", type=str,
                        default="/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete.pkl",
                        help="Dataset path")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("-epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("-device", type=str, default="cuda:0", help="Device (cuda:0/cpu)")
    parser.add_argument("-model_save_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_final_2",
                        help="Directory to save best models")
    parser.add_argument("-log_dir", type=str, default="logfile_model2_2_2", help="Log directory")
    parser.add_argument("-results_dir", type=str, default="results_model2_2_2", help="Results directory")
    parser.add_argument("-patience", type=int, default=20, help="EarlyStopping patience（测试用2）")
    parser.add_argument("-min_delta", type=float, default=0.001, help="EarlyStopping min_delta（测试用1.0）")
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    cv_data = CVDatasets(fpath=args.data_fpath, batch_size=args.batch_size)

    sbt_compress = th.nn.Linear(935, 256).to(args.device)
    sbt_compress.weight.data.normal_(0, 0.01)
    sbt_compress.bias.data.fill_(0.0)

    all_pred_label = []
    all_indices = []

    for fold in range(10):
        print(f"\n=== Start Training Fold {fold} ===")
        train_loader, valid_idx, valid_loader = cv_data.get_dataloader(fold)

        model = Model_Regression().to(args.device)
        # print("模型参数名列表：", list(model.state_dict().keys()))
        optimizer = th.optim.Adam(
            list(model.parameters()) + list(sbt_compress.parameters()),
            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5
        )

        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        # 核心修改1：初始化最佳参数时用深拷贝
        best_model_state = copy.deepcopy(model.state_dict())
        best_compress_state = copy.deepcopy(sbt_compress.state_dict())

        # 打印初始化哈希
        init_model_hash = hash(str(best_model_state["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
        init_compress_hash = hash(str(best_compress_state["weight"].cpu().numpy().tobytes()))
        # print(f"Fold {fold} - 初始化最佳模型参数：")
        # print(f"  模型核心参数哈希：{init_model_hash}")
        # print(f"  底物压缩层参数哈希：{init_compress_hash}")

        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_header = [
            "epoch", "train_pcc", "train_scc", "train_r2", "train_rmse", "train_loss",
            "valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_loss",
            "is_best_model", "current_best_epoch"
        ]
        record_data = []

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            # 训练并打印训练后的参数哈希（验证是否更新）
            train_metrics = custom_train_epoch(
                model=model, data_loader=train_loader, optimizer=optimizer,
                device=args.device, sbt_compress=sbt_compress, epoch=epoch  # 传入epoch用于打印
            )
            valid_pred, valid_label, valid_metrics = custom_eval_epoch(
                model=model, data_loader=valid_loader, device=args.device,
                sbt_compress=sbt_compress
            )

            is_best, need_stop = early_stopping.check(epoch, valid_metrics[-1])
            if is_best:
                # 核心修改2：更新最佳参数时用深拷贝
                old_model_hash = hash(str(best_model_state["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
                old_compress_hash = hash(str(best_compress_state["weight"].cpu().numpy().tobytes()))

                best_model_state = copy.deepcopy(model.state_dict())  # 强制深拷贝
                best_compress_state = copy.deepcopy(sbt_compress.state_dict())  # 强制深拷贝

                new_model_hash = hash(str(best_model_state["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
                new_compress_hash = hash(str(best_compress_state["weight"].cpu().numpy().tobytes()))
                # print(f"Fold {fold} - Epoch {epoch}: 触发最佳模型更新！")
                # print(f"  模型参数哈希：{old_model_hash} → {new_model_hash}（深拷贝后）")
                # print(f"  压缩层参数哈希：{old_compress_hash} → {new_compress_hash}（深拷贝后）")

                model_save_path = os.path.join(args.model_save_dir, f"fold{fold}_best_params.pth")
                th.save({
                    "model_state": best_model_state,
                    "compress_state": best_compress_state,
                    "best_epoch": epoch,
                    "valid_rmse": valid_metrics[3],
                    "valid_loss": valid_metrics[4]
                }, model_save_path)
                # print(f"  最佳模型已保存至：{model_save_path}")
                # print(f"  保存的最佳epoch：{epoch}，对应RMSE：{valid_metrics[3]:.4f}")

            current_best_epoch = early_stopping.best_epoch
            log_entry = np.concatenate([
                np.array([epoch]),
                train_metrics,
                valid_metrics,
                np.array([1 if is_best else 0]),
                np.array([current_best_epoch])
            ])
            record_data.append(log_entry)

            write_logfile(
                epoch=epoch,
                record_data=record_data,
                logfile=os.path.join(args.log_dir, f"logfile_fold{fold}.csv")
            )

            if need_stop:
                print(f"Fold {fold} - Early Stopping Triggered! Stop at Epoch {epoch}")
                # print(f"  早停时记录的最佳epoch：{current_best_epoch}")
                # print(
                #     f"  早停时最佳模型参数哈希：{hash(str(best_model_state['model_fc.convs.0.weight'].cpu().numpy().tobytes()))}")
                break

        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(args.log_dir, f"time_fold{fold}.dat"), 'w') as f:
            f.write(f"Fold {fold} Training Time\nStart: {start_time}\nEnd: {end_time}")

        # 加载前哈希对比
        before_load_model_hash = hash(str(model.state_dict()["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
        best_model_hash = hash(str(best_model_state["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
        # print(f"\nFold {fold} - 加载最佳模型前：")
        # print(f"  当前模型参数哈希：{before_load_model_hash}")
        # print(f"  最佳模型参数哈希：{best_model_hash}")
        # print(f"  两者是否一致？{before_load_model_hash == best_model_hash}")

        # 加载最佳模型
        model.load_state_dict(best_model_state)
        sbt_compress.load_state_dict(best_compress_state)

        # 加载后哈希对比
        after_load_model_hash = hash(str(model.state_dict()["model_fc.convs.0.weight"].cpu().numpy().tobytes()))
        # print(f"Fold {fold} - 加载最佳模型后：")
        # print(f"  模型参数哈希：{after_load_model_hash}")
        # print(f"  与最佳模型是否一致？{after_load_model_hash == best_model_hash}")

        # 重新验证
        final_valid_pred, final_valid_label, final_valid_metrics = custom_eval_epoch(
            model=model, data_loader=valid_loader, device=args.device,
            sbt_compress=sbt_compress
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

        # 对比模型文件指标
        model_save_path = os.path.join(args.model_save_dir, f"fold{fold}_best_params.pth")
        saved_ckpt = th.load(model_save_path)
        saved_rmse = saved_ckpt["valid_rmse"]
        # print(f"Fold {fold} - 模型文件中保存的最优RMSE：{saved_rmse:.4f}")
        # print(f"  重新验证的RMSE：{final_valid_metrics[3]:.4f}")
        # print(f"  指标差异：{abs(final_valid_metrics[3] - saved_rmse):.4f}")

        # 保存结果
        with open(os.path.join(args.log_dir, f"logfile_fold{fold}.csv"), 'a') as f:
            f.write(f"\n# {best_result_log.replace(chr(10), chr(10) + '# ')}")
        out_results(final_valid_metrics, os.path.join(args.results_dir, f"results_fold{fold}.csv"))

        all_pred_label.append(
            np.concatenate([final_valid_pred.reshape(-1, 1), final_valid_label.reshape(-1, 1)], axis=1)
        )
        all_indices.extend(valid_idx)
        print(f"=== Finish Training Fold {fold} ===")

    # 汇总全局结果
    all_pred_label = np.concatenate(all_pred_label, axis=0)
    final_pred_df = pd.DataFrame(
        all_pred_label, index=all_indices, columns=["pred_logkcat", "label_logkcat"]
    )
    final_pred_df.to_csv("final_pred_label_model2_2_2.csv")

    final_pcc, final_scc, final_r2, final_rmse = evaluate(all_pred_label[:, 1], all_pred_label[:, 0])
    final_metrics = pd.DataFrame(
        np.array([[final_pcc, final_scc, final_r2, final_rmse]]),
        columns=["PCC", "SCC", "R2", "RMSE"]
    )
    final_metrics.to_csv("final_results_model2_2_2.csv", index=False)

    print(f"\n全局最终结果：PCC={final_pcc:.4f}, SCC={final_scc:.4f}, R2={final_r2:.4f}, RMSE={final_rmse:.4f}")