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

# 根据需要选择模型（二选一）
# 选项1：纯冻结版
from model_freeze import Model_Regression, ActivityModel_Freeze as ActivityModel
# 选项2：部分微调版（注释上面，打开下面）
# from model_finetune import Model_Regression, ActivityModel_Finetune as ActivityModel


# ------------------------- 数据集类（支持10折） -------------------------
class Mydatasets(Dataset):
    def __init__(self, ezy_feats, sbt_feats, labels):
        self.ezy_feats = ezy_feats  # list of (seq_len, 1024)
        self.sbt_feats = sbt_feats  # (n_samples, 935)
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
        data_df = pd.read_pickle(fpath)[["new_ezy_feat", "sbt_feat", "log10_kcat", "fold"]].copy()
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


# ------------------------- 训练/验证函数（精简输出） -------------------------
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

        # 收集结果（用于计算指标）
        with th.no_grad():
            total_loss += loss.item()
            y_label.append(labels.cpu().numpy())
            y_fusion_pred.append(final_logkcat.cpu().numpy())
            y_left_pred.append(pred_left_logkcat.cpu().numpy())
            y_right_pred.append(pred_right_logkcat.cpu().numpy())

    # 计算指标（不打印，只返回）
    y_label = np.concatenate(y_label)
    y_fusion_pred = np.concatenate(y_fusion_pred)
    y_left_pred = np.concatenate(y_left_pred)
    y_right_pred = np.concatenate(y_right_pred)

    fusion_pcc, fusion_scc, fusion_r2, fusion_rmse = evaluate(y_label, y_fusion_pred)
    left_pcc, left_scc, left_r2, left_rmse = evaluate(y_label, y_left_pred)
    right_pcc, right_scc, right_r2, right_rmse = evaluate(y_label, y_right_pred)
    avg_loss = total_loss / len(data_loader)

    return np.array([
        fusion_pcc, fusion_scc, fusion_r2, fusion_rmse,  # 融合指标（0-4）
        left_pcc, left_scc, left_r2, left_rmse,          # 左路指标（5-8）
        right_pcc, right_scc, right_r2, right_rmse,      # 右路指标（9-12）
        avg_loss                                         # 平均损失（13）
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

    # 计算指标（不打印，只返回）
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
            fusion_pcc, fusion_scc, fusion_r2, fusion_rmse,  # 融合指标（0-4）
            left_pcc, left_scc, left_r2, left_rmse,          # 左路指标（5-8）
            right_pcc, right_scc, right_r2, right_rmse,      # 右路指标（9-12）
            avg_loss                                         # 平均损失（13）
        ])
    )


# ------------------------- 10折主训练逻辑 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10折训练 - 输出左路/右路/融合结果（精简输出）",
                                     formatter_class=RawDescriptionHelpFormatter)
    # 数据路径
    parser.add_argument("-data_fpath", type=str,
                        default="/mnt/usb3/code/wm/data/kcat_data/kcat-data_feats_complete.pkl",
                        help="kcat数据集路径")
    # 预训练模型路径
    parser.add_argument("-kcat_km_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/kcat_km/model3_1",
                        help="kcat/km预训练模型目录")
    parser.add_argument("-km_model_dir", type=str,
                        default="/mnt/usb/code/wm/catapro/models/km/km_model2_1",
                        help="Km预训练模型目录")
    # 输出路径（10折统一目录，按fold区分）
    parser.add_argument("-log_dir", type=str, default="logfile_model_freeze", help="日志目录（含左右路）")
    parser.add_argument("-results_dir", type=str, default="results_model_freeze", help="结果目录（含左右路）")
    parser.add_argument("-model_save_dir", type=str, default="/mnt/usb/code/wm/catapro/models/kcat/models_model_freeze", help="模型保存目录")
    # 超参数
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-5, help="学习率（冻结版1e-5，微调版5e-6）")
    parser.add_argument("-epochs", type=int, default=150, help="最大epoch")
    parser.add_argument("-alpha", type=float, default=0.5, help="左路权重")
    parser.add_argument("-device", type=str, default="cuda:0", help="设备（cuda:0/cpu）")
    # 早停参数
    parser.add_argument("-patience", type=int, default=20, help="早停patience")
    parser.add_argument("-min_delta", type=float, default=0.001, help="早停最小变化量")
    args = parser.parse_args()

    # 创建10折统一输出目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    # 加载数据集（支持10折）
    print(f"加载kcat数据集：{args.data_fpath}")
    cv_data = CVDatasets(fpath=args.data_fpath, batch_size=args.batch_size)
    print(f"数据集加载完成，总样本数：{len(cv_data.labels)}")

    # 10折训练循环
    for fold in range(10):
        print(f"\n" + "=" * 80)
        print(f"[Fold {fold}/{9}] 开始训练")
        print("=" * 80)

        # 1. 加载当前fold的预训练模型和压缩器（仅右路）
        kcat_km_ckpt_path = os.path.join(args.kcat_km_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(kcat_km_ckpt_path):
            raise FileNotFoundError(f"kcat/km模型不存在：{kcat_km_ckpt_path}")
        kcat_km_ckpt = th.load(kcat_km_ckpt_path, map_location=args.device)

        kcat_km_model = Model_Regression().to(args.device)
        kcat_km_model.load_state_dict(kcat_km_ckpt["model_state"])
        kcat_km_compress_state = kcat_km_ckpt["compress_state"]

        km_ckpt_path = os.path.join(args.km_model_dir, f"fold{fold}_best_params.pth")
        if not os.path.exists(km_ckpt_path):
            raise FileNotFoundError(f"Km模型不存在：{km_ckpt_path}")
        km_ckpt = th.load(km_ckpt_path, map_location=args.device)

        km_model = Model_Regression().to(args.device)
        km_model.load_state_dict(km_ckpt["model_state"])
        km_compress_state = km_ckpt["compress_state"]

        # 2. 初始化融合模型（左路随机初始化，不加载预训练）
        model = ActivityModel(
            kcat_km_model=kcat_km_model,
            Km_model=km_model,
            kcat_km_compress_state=kcat_km_compress_state,
            km_compress_state=km_compress_state,
            alpha=args.alpha,
            device=args.device
        ).to(args.device)

        # 3. 模型参数配置（根据模型类型选择）
        if "ActivityModel_Freeze" in str(ActivityModel):
            # 纯冻结版：固定右路所有参数+压缩器，左路随机初始化并可训练
            for param in kcat_km_model.parameters():
                param.requires_grad = False
            for param in km_model.parameters():
                param.requires_grad = False
            for param in model.compressor_kcat_km.parameters():
                param.requires_grad = False
            for param in model.compressor_km.parameters():
                param.requires_grad = False
            # 校验左路可训练参数数量（确保左路在训练）
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Fold {fold}] 纯冻结版：固定右路所有参数+压缩器，左路随机初始化可训练（可训练参数数：{trainable_params}），保留detach()")
        else:
            # 部分微调版：解冻右路顶层参数
            for name, param in kcat_km_model.named_parameters():
                if "fc1" in name or "fc2" in name or "classifier_layer" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in km_model.named_parameters():
                if "fc1" in name or "fc2" in name or "classifier_layer" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for param in model.compressor_kcat_km.parameters():
                param.requires_grad = False
            for param in model.compressor_km.parameters():
                param.requires_grad = False
            print(f"[Fold {fold}] 部分微调版：解冻右路顶层参数，固定压缩器，去掉detach()")

        # 4. 优化器
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

        # 6. 保存压缩器初始权重（校验用，仅纯冻结版）
        if "ActivityModel_Freeze" in str(ActivityModel):
            kcat_km_compress_init = copy.deepcopy({
                "weight": model.compressor_kcat_km.weight.data,
                "bias": model.compressor_kcat_km.bias.data
            })
            km_compress_init = copy.deepcopy({
                "weight": model.compressor_km.weight.data,
                "bias": model.compressor_km.bias.data
            })

        # 7. 获取当前fold的加载器
        train_loader, valid_idx, valid_loader = cv_data.get_dataloader(fold=fold)

        # 8. 训练循环（仅输出Epoch序号）
        for epoch in range(args.epochs):
            # 只打印Epoch序号，不打印详细指标
            print(f"[Epoch {epoch + 1}/{args.epochs}]")

            # 训练
            train_metrics = custom_train_epoch(model, train_loader, optimizer, args.device)

            # 验证
            valid_fusion_pred, valid_left_pred, valid_right_pred, valid_label, valid_metrics = custom_eval_epoch(
                model, valid_loader, args.device
            )

            # # 压缩器权重校验（纯冻结版，每10轮校验）
            # if (epoch + 1) % 10 == 0 and "ActivityModel_Freeze" in str(ActivityModel):
            #     kcat_km_weight_error = th.mean(
            #         th.abs(model.compressor_kcat_km.weight.data - kcat_km_compress_init["weight"]))
            #     if kcat_km_weight_error > 1e-6:
            #         raise RuntimeError(f"[Fold {fold}] 警告！kcat/km压缩器权重被意外修改（误差：{kcat_km_weight_error:.8f}）")
            #     km_weight_error = th.mean(th.abs(model.compressor_km.weight.data - km_compress_init["weight"]))
            #     if km_weight_error > 1e-6:
            #         raise RuntimeError(f"[Fold {fold}] 警告！Km压缩器权重被意外修改（误差：{km_weight_error:.8f}）")
            #     print(f"[Epoch {epoch + 1}] 右路压缩器权重冻结校验通过")

            # 记录日志（修正左路/右路指标索引，确保列名与数值对应）
            log_entry = np.concatenate([
                np.array([epoch]),
                # 融合指标（0-3：PCC/SCC/R2/RMSE）
                np.array([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]]),
                # 左路指标（4-7：PCC/SCC/R2/RMSE）- 修正前是5-8，现在对齐返回顺序
                np.array([valid_metrics[4], valid_metrics[5], valid_metrics[6], valid_metrics[7]]),
                # 右路指标（8-11：PCC/SCC/R2/RMSE）- 修正前是9-12，现在对齐返回顺序
                np.array([valid_metrics[8], valid_metrics[9], valid_metrics[10], valid_metrics[11]]),
                # 训练集融合指标（0-3：PCC/SCC/R2/RMSE）
                np.array([train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3]]),
                # 损失（训练损失：12，验证损失：12）
                np.array([train_metrics[12], valid_metrics[12]]),
                # 早停相关
                np.array([1 if early_stopping.is_bestmodel else 0]),
                np.array([early_stopping.best_epoch])
            ])
            log_records.append(log_entry)

            # 写入日志文件（每个fold单独的日志，表头与数值严格对应）
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

            # 早停检查 + 保存最优模型
            is_best, need_stop = early_stopping.check(epoch, valid_metrics[12])  # 按验证损失判断
            if is_best:
                best_model_state = copy.deepcopy(model.state_dict())
                # 保存最优模型（每个fold单独保存，包含核心参数）
                best_model_path = os.path.join(args.model_save_dir, f"fold{fold}_best_model.pth")
                th.save({
                    "model_state": best_model_state,
                    "epoch": epoch,
                    "valid_loss": valid_metrics[12],
                    "valid_fusion_pcc": valid_metrics[0],
                    "valid_left_pcc": valid_metrics[4],  # 修正：左路PCC索引4（之前是5）
                    "valid_right_pcc": valid_metrics[8],  # 修正：右路PCC索引8（之前是9）
                    "alpha": args.alpha,
                    "lr": args.lr
                }, best_model_path)
                print(f"[Epoch {epoch + 1}] 保存最优模型至：{best_model_path}")

            if need_stop:
                print(f"早停触发，终止于Epoch {epoch + 1}")
                break

        # 9. 加载最优模型重新验证（确保结果可靠）
        model.load_state_dict(best_model_state)
        final_fusion_pred, final_left_pred, final_right_pred, final_label, final_metrics = custom_eval_epoch(
            model, valid_loader, args.device
        )

        # 10. 解析最终指标（修正索引，确保所有指标准确）
        final_fusion_pcc = final_metrics[0]
        final_fusion_scc = final_metrics[1]
        final_fusion_r2 = final_metrics[2]
        final_fusion_rmse = final_metrics[3]
        final_left_pcc = final_metrics[4]  # 修正：左路PCC（原5→4）
        final_left_scc = final_metrics[5]  # 修正：左路SCC（原6→5）
        final_left_r2 = final_metrics[6]  # 修正：左路R2（原7→6）
        final_left_rmse = final_metrics[7]  # 修正：左路RMSE（原8→7）
        final_right_pcc = final_metrics[8]  # 修正：右路PCC（原9→8）
        final_right_scc = final_metrics[9]  # 修正：右路SCC（原10→9）
        final_right_r2 = final_metrics[10]  # 修正：右路R2（原11→10）
        final_right_rmse = final_metrics[11]  # 修正：右路RMSE（原12→11）

        # 11. 保存当前fold最终指标汇总（可直接用于后续10折均值统计）
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

        # 12. 保存详细预测结果（包含残差，方便后续误差分析）
        final_pred_df = pd.DataFrame({
            "fold": [fold] * len(valid_idx),
            "sample_idx": valid_idx,
            "label_log10_kcat": final_label,
            "pred_fusion_log10_kcat": final_fusion_pred,
            "pred_left_log10_kcat": final_left_pred,
            "pred_right_log10_kcat": final_right_pred,
            "fusion_residual": final_fusion_pred - final_label,  # 融合模型残差
            "left_residual": final_left_pred - final_label,  # 左路残差
            "right_residual": final_right_pred - final_label  # 右路残差
        })
        final_pred_path = os.path.join(args.results_dir, f"fold{fold}_final_pred_detail.csv")
        final_pred_df.to_csv(final_pred_path, index=False, float_format="%.6f")

        # 13. 打印当前fold最终结果汇总（清晰展示性能）
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

        # 14. 输出当前fold核心文件路径（方便后续查找）
        print(f"[Fold {fold}] 核心文件保存路径：")
        print(f"- 训练日志：{os.path.join(args.log_dir, f'fold{fold}_log.csv')}")
        print(f"- 最终指标汇总：{final_results_path}")
        print(f"- 详细预测结果：{final_pred_path}")
        print(f"- 最优模型权重：{best_model_path}")
        print("=" * 80)

    # 10折训练全部完成后，生成全局汇总提示
    print(f"\n" + "=" * 80)
    print("🎉 10折交叉训练全部完成！")
    print("=" * 80)
    print(f"📁 所有结果文件汇总：")
    print(f"- 训练日志目录：{args.log_dir}（10个fold分别的训练日志）")
    print(f"- 指标/预测结果目录：{args.results_dir}（含每个fold的最终指标和详细预测）")
    print(f"- 最优模型目录：{args.model_save_dir}（10个fold的最优模型权重）")
    print(f"\n💡 后续操作建议：")
    print(f"1. 运行10折结果统计脚本，计算融合/左路/右路的PCC/SCC/R2/RMSE均值±标准差")
    print(f"2. 对比纯冻结版与部分微调版的10折均值，选择最优模型架构")
    print(f"3. 基于详细预测结果，分析模型误差分布（如残差较大的样本特征）")
    print("=" * 80)