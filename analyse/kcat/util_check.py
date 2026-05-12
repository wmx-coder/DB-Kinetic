import os
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ------------------------- 评估指标函数（完全不变，直接复用） -------------------------
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    fsp = y_pred - np.mean(y_pred)
    fst = y_true - np.mean(y_true)

    devP = np.std(y_pred)
    devT = np.std(y_true)

    return np.mean(fsp * fst) / (devP * devT) if (devP * devT) != 0 else 0.0  # 避免除零错误


def SCC(y_true, y_pred):
    X_rank = rankdata(y_pred)
    Y_rank = rankdata(y_true)

    return PCC(Y_rank, X_rank)


def R_2(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - y_true_mean))

    return 1 - numerator / denominator if denominator != 0 else 0.0  # 避免除零错误


def evaluate(y_true, y_pred):
    _pcc = PCC(y_true, y_pred)
    _rmse = RMSE(y_true, y_pred)
    _r2 = R_2(y_true, y_pred)
    _scc = SCC(y_true, y_pred)

    return _pcc, _scc, _r2, _rmse


# ------------------------- 结果保存函数（修改：适配三路指标） -------------------------
def out_results(values, file_):
    """
    保存最终结果（融合+左路+右路）
    values: 包含三路指标的数组，格式为 [融合PCC, 融合SCC, 融合R2, 融合RMSE, 左路PCC, 左路SCC, 左路R2, 左路RMSE, 右路PCC, 右路SCC, 右路R2, 右路RMSE, 验证损失]
    """
    columns = [
        "valid_fusion_pcc", "valid_fusion_scc", "valid_fusion_r2", "valid_fusion_rmse",
        "valid_left_pcc", "valid_left_scc", "valid_left_r2", "valid_left_rmse",
        "valid_right_pcc", "valid_right_scc", "valid_right_r2", "valid_right_rmse",
        "valid_loss"
    ]
    # 确保输入维度匹配（13个元素）
    if len(values) != 13:
        raise ValueError(f"out_results 输入维度错误！需13个元素，实际{len(values)}个")
    df = pd.DataFrame(values.reshape(1, -1), columns=columns)
    df.to_csv(file_, float_format="%.5f", index=False)


# ------------------------- 删除原 write_logfile 函数 -------------------------
# 原因：主代码中已经实现了更详细的日志记录（包含三路指标），无需重复，避免冲突


# ------------------------- EarlyStopping 类（完全不变，直接复用） -------------------------
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


# ------------------------- RMSELoss 类（完全不变，直接复用） -------------------------
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_true, y_pred):
        loss = th.sqrt(self.mse(y_true, y_pred) + self.eps)
        return loss


rmse_loss = RMSELoss()  # 实例化，供主代码调用