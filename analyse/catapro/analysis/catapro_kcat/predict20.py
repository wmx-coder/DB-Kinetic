import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
from model import *
from act_model import KcatModel as _KcatModel
from torch.utils.data import Dataset
from argparse import RawDescriptionHelpFormatter
import argparse

import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)


class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def inference_single_data(model, ezy_feats, sbt_feats, device="cuda:0"):
    """单条数据预测函数（修复tuple输出问题）"""
    model.eval()
    with th.no_grad():
        pred = model(ezy_feats, sbt_feats)
        # 关键修复：如果输出是tuple，取最后一个元素（或第一个，根据模型定义）
        if isinstance(pred, tuple):
            pred = pred[-1]  # 优先取最后一个元素（和ActivityModel保持一致）
        pred = pred.cpu().numpy()
    return pred


if __name__ == "__main__":
    d = "RUN CATAPRO (only kcat prediction, use pre-extracted feats) ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_fpath", type=str, default="/mnt/usb3/wmx/analyse/kcat-data_filtered_N20.pkl",
                        help="Input (.pkl). The path of enzyme kcat data file.")
    parser.add_argument("-model_dpath", type=str, default="/mnt/usb1/wmx/catapro/models/catapro/models",
                        help="Input. The path of saved models.")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    parser.add_argument("-out_fpath", type=str, default="./N20.pkl",
                        help="Output. Path to save data with prediction (current dir by default).")
    args = parser.parse_args()

    # 1. 加载原始数据并处理fold类型
    inp_df = pd.read_pickle(args.inp_fpath)
    # 检查必要列（新增ezy_feat、sbt_feat）
    required_cols = ["fold", "ezy_feat", "sbt_feat"]
    for col in required_cols:
        if col not in inp_df.columns:
            raise ValueError(f"原始数据缺少必要列：{col}")
    inp_df["fold"] = inp_df["fold"].astype(int)  # 确保fold是整数

    # 2. 初始化路径
    model_dpath = args.model_dpath
    kcat_model_dpath = f"{model_dpath}/kcat_models"
    device = args.device

    # 3. 预加载所有fold的kcat模型
    model_cache = {}
    unique_folds = inp_df["fold"].unique()
    for fold in unique_folds:
        try:
            kcat_model = _KcatModel(device=device)
            kcat_model.load_state_dict(th.load(f"{kcat_model_dpath}/{fold}_bestmodel.pth", map_location=device))
            model_cache[fold] = kcat_model
            print(f"成功加载fold {fold} 的kcat模型")
        except Exception as e:
            raise ValueError(f"加载fold {fold} 的kcat模型失败：{e}")

    # 4. 遍历预测（仅kcat，复用已有特征）
    pred_kcat_list = []
    total_rows = len(inp_df)
    current_row = 0  # 修复进度提示的计数器
    for idx, row in inp_df.iterrows():
        try:
            current_row += 1
            if current_row % 100 == 0:
                print(f"处理进度：{current_row}/{total_rows}")

            fold = row["fold"]
            # 直接读取已有特征，无需重新提取
            ezy_feat = row["ezy_feat"]  # 酶特征
            sbt_feat = row["sbt_feat"]  # 底物特征

            # 转换为tensor并适配模型输入格式
            ezy_feats = th.from_numpy(ezy_feat).to(th.float32).to(device).unsqueeze(0)  # 增加batch维度 (1, 1024)
            sbt_feats = th.from_numpy(sbt_feat).to(th.float32).to(device).unsqueeze(0)  # 增加batch维度

            # 预测kcat
            kcat_model = model_cache[fold]
            pred_kcat = inference_single_data(kcat_model, ezy_feats, sbt_feats, device)

            pred_kcat_list.append(pred_kcat[0][0])

        except Exception as e:
            print(f"处理第{current_row}行（索引{idx}）时出错：{e}")
            pred_kcat_list.append(np.nan)

    # 5. 新增列并保存
    inp_df["pred_log10[kcat(s^-1)]"] = pred_kcat_list
    inp_df.to_pickle(args.out_fpath)
    csv_out_path = args.out_fpath.replace(".pkl", ".csv")
    # CSV保存时跳过特征列（避免数组字符串化）
    non_array_cols = [col for col in inp_df.columns if not isinstance(inp_df[col].iloc[0], np.ndarray)]
    inp_df[non_array_cols].to_csv(csv_out_path, index=False)

    print(f"\n预测完成！结果保存至：")
    print(f"- Pickle: {args.out_fpath}")
    print(f"- CSV: {csv_out_path}")
    print(f"- 总计处理 {total_rows} 行，出错 {np.isnan(pred_kcat_list).sum()} 行")