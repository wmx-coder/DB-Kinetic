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
class EarlyStopping(object):
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = float('inf')  # 初始化为无穷大，确保第一轮一定更新
        self.count_epoch = 0
        self.stop = False
        self.is_bestmodel = False
        self.best_epoch = 0

    def check(self, epoch, cur_loss):
        # 取消 epoch==0 的特殊处理，统一按损失是否下降判断
        if cur_loss < self.min_loss - self.min_delta:  # 损失下降超阈值，才视为最佳
            self.min_loss = cur_loss
            self.count_epoch = 0
            self.is_bestmodel = True
            self.best_epoch = epoch  # 记录当前最优epoch
        else:
            self.count_epoch += 1
            self.is_bestmodel = False
        if self.count_epoch == self.patience:
            self.stop = True
        return self.is_bestmodel, self.stop

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_true, y_pred):
        loss = th.sqrt(self.mse(y_true, y_pred) + self.eps)
        return loss


rmse_loss = RMSELoss()