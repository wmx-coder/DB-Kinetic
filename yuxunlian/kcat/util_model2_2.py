import os
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    fsp = y_pred - np.mean(y_pred)
    fst = y_true - np.mean(y_true)

    devP = np.std(y_pred)
    devT = np.std(y_true)

    return np.mean(fsp * fst) / (devP * devT)


def SCC(y_true, y_pred):
    X_rank = rankdata(y_pred)
    Y_rank = rankdata(y_true)

    return PCC(Y_rank, X_rank)


def R_2(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - y_true_mean))

    return 1 - numerator / denominator


def evaluate(y_true, y_pred):
    _pcc = PCC(y_true, y_pred)
    _rmse = RMSE(y_true, y_pred)
    _r2 = R_2(y_true, y_pred)
    _scc = SCC(y_true, y_pred)

    return _pcc, _scc, _r2, _rmse


def out_results(values, file_):
    columns = ["valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_loss"]
    df = pd.DataFrame(values.reshape(1, -1), columns=columns)
    df.to_csv(file_, float_format="%.5f")


def write_logfile(epoch, record_data, logfile):
    # 1. 定义完整列名（包含新增的"是否最佳模型"和"当前最佳epoch"）
    columns = [
        "epoch", "train_pcc", "train_scc", "train_r2", "train_rmse", "train_loss",
        "valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_loss",
        "is_best_model"  , "current_best_epoch"# 新增字段，与主代码日志对应
    ]

    # 2. 处理数据格式（确保与列数匹配）
    values = np.array(record_data).reshape(epoch + 1, -1)  # 行数=epoch+1，列数=13（与columns对应）

    # 3. 首次写入时删除旧文件，重新创建；后续直接覆盖（保持文件始终是完整日志）
    if epoch == 0 and os.path.exists(logfile):
        os.remove(logfile)

    # 4. 写入CSV（保留4位小数，索引用epoch值）
    df = pd.DataFrame(values, index=range(epoch + 1), columns=columns)
    df.to_csv(logfile, float_format="%.4f", index=False)  # 不写入行索引（避免重复）


# utils.py 中完整的 EarlyStopping 类
# util_model2.py 完整修改后的 EarlyStopping 类
class EarlyStopping(object):
    def __init__(self, patience, min_delta, monitor="valid_rmse"):
        """
        Args:
            patience: 连续无有效提升的轮次阈值
            min_delta: 有效提升的最小阈值（如0.001）
            monitor: 监控的核心指标，可选 "valid_rmse"（默认）、"valid_loss"、"valid_pcc"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor  # 新增：指定监控的指标

        # 根据指标类型初始化“最佳值”和“有效提升判定逻辑”
        if self.monitor in ["valid_rmse", "valid_loss"]:
            # RMSE/损失：越小越好，初始值设为无穷大
            self.best_value = float('inf')
            # 判定逻辑：当前值 < 最佳值 - 最小阈值 → 有效提升
            self.is_better = lambda curr: curr < self.best_value - self.min_delta
        elif self.monitor in ["valid_pcc", "valid_scc", "valid_r2"]:
            # PCC/SCC/R2：越大越好，初始值设为负无穷
            self.best_value = -float('inf')
            # 判定逻辑：当前值 > 最佳值 + 最小阈值 → 有效提升
            self.is_better = lambda curr: curr > self.best_value + self.min_delta

        self.count_epoch = 0  # 连续无提升计数器
        self.stop = False  # 是否终止训练
        self.is_bestmodel = False  # 当前是否为最佳模型
        self.best_epoch = 0  # 最佳模型对应的epoch

    def check(self, epoch, valid_metrics):
        """
        输入完整验证指标，判断是否为最佳模型/是否早停
        Args:
            epoch: 当前训练轮次（0开始）
            valid_metrics: 完整验证指标列表，顺序为 [pcc, scc, r2, rmse, loss]
        Returns:
            is_bestmodel: 是否为最佳模型
            stop: 是否需要终止训练
        """
        # 1. 根据监控指标，从valid_metrics中提取当前值
        metric_index = {
            "valid_pcc": 0,
            "valid_scc": 1,
            "valid_r2": 2,
            "valid_rmse": 3,
            "valid_loss": 4
        }
        current_value = valid_metrics[metric_index[self.monitor]]  # 提取RMSE值

        # 2. 判断是否为有效提升
        if self.is_better(current_value):
            self.best_value = current_value  # 更新最佳值
            self.count_epoch = 0  # 重置无提升计数器
            self.is_bestmodel = True  # 标记为最佳模型
            self.best_epoch = epoch  # 更新最佳epoch
        else:
            self.count_epoch += 1  # 无提升，计数器+1
            self.is_bestmodel = False  # 不标记为最佳

        # 3. 判断是否触发早停
        if self.count_epoch >= self.patience:
            self.stop = True

        return self.is_bestmodel, self.stop

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_true, y_pred):
        loss = th.sqrt(self.mse(y_true, y_pred) + self.eps)
        return loss


rmse_loss = RMSELoss()