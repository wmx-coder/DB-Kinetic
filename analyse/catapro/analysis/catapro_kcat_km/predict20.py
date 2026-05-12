import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
from model import *
from act_model import ActivityModel
from torch.utils.data import Dataset
from argparse import RawDescriptionHelpFormatter
import argparse

import logging
from transformers import logging as hf_logging

# 抑制冗余日志
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)


class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def inference_single_act(model, ezy_feats, sbt_feats, device="cuda:0"):
    """单条数据预测act模型（适配ActivityModel输出格式）"""
    model.eval()
    with th.no_grad():
        pred = model(ezy_feats, sbt_feats)
        # ActivityModel返回列表，取最后一个元素（与原代码逻辑一致）
        if isinstance(pred, list) or isinstance(pred, tuple):
            pred = pred[-1]
        pred = pred.cpu().numpy()
    return pred


if __name__ == "__main__":
    d = "RUN CATAPRO (only ActivityModel prediction for kcat-km dataset) ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_fpath", type=str, default="/mnt/usb3/wmx/analyse/kcat-km_data_filtered_N20_with_ezy_feat.pkl",
                        help="Input (.pkl). The path of kcat-km data file.")
    parser.add_argument("-model_dpath", type=str, default="/mnt/usb1/wmx/catapro/models/catapro/models",
                        help="Input. The path of saved models.")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    parser.add_argument("-out_fpath", type=str, default="./kcat-km_N20_act_pred.pkl",
                        help="Output. Path to save data with act prediction (current dir by default).")
    args = parser.parse_args()

    # 1. 加载原始数据并校验
    inp_df = pd.read_pickle(args.inp_fpath)
    # 检查必要列（fold + 特征列）
    required_cols = ["fold", "ezy_feat", "sbt_feat"]
    missing_cols = [col for col in required_cols if col not in inp_df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必要列：{missing_cols}，请检查数据格式")

    # 确保fold为整数（避免5.0等浮点数问题）
    inp_df["fold"] = inp_df["fold"].astype(int)
    total_rows = len(inp_df)
    print(f"成功加载kcat-km数据集，共 {total_rows} 条数据")

    # 2. 初始化路径和设备
    model_dpath = args.model_dpath
    act_model_dpath = f"{model_dpath}/act_models"  # 仅act模型路径
    device = args.device

    # 3. 预加载所有fold对应的act模型（避免重复加载）
    model_cache = {}
    unique_folds = inp_df["fold"].unique()
    for fold in unique_folds:
        try:
            act_model = ActivityModel(device=device)
            act_model.load_state_dict(th.load(f"{act_model_dpath}/{fold}_bestmodel.pth", map_location=device))
            model_cache[fold] = act_model
            print(f"成功加载fold {fold} 的ActivityModel")
        except Exception as e:
            raise ValueError(f"加载fold {fold} 的act模型失败：{e}")

    # 4. 遍历数据预测act结果
    pred_act_list = []
    current_row = 0  # 进度计数器（修复索引混乱问题）
    for idx, row in inp_df.iterrows():
        try:
            current_row += 1
            # 进度提示（每100行打印一次）
            if current_row % 100 == 0:
                print(f"处理进度：{current_row}/{total_rows}")

            # 获取fold和已有特征
            fold = row["fold"]
            ezy_feat = row["ezy_feat"]
            sbt_feat = row["sbt_feat"]

            # 特征格式适配：转为tensor + 增加batch维度 + 送入设备
            ezy_feats = th.from_numpy(ezy_feat).to(th.float32).to(device).unsqueeze(0)
            sbt_feats = th.from_numpy(sbt_feat).to(th.float32).to(device).unsqueeze(0)

            # 调用对应fold的act模型预测
            act_model = model_cache[fold]
            pred_act = inference_single_act(act_model, ezy_feats, sbt_feats, device)

            # 保存预测结果（取标量值）
            pred_act_list.append(pred_act[0][0])

        except Exception as e:
            print(f"处理第{current_row}行（索引{idx}）时出错：{e}")
            pred_act_list.append(np.nan)

    # 5. 新增act预测列并保存
    inp_df["pred_log10[kcat/Km(s^-1mM^-1)]"] = pred_act_list

    # 保存完整pkl文件（保留所有原始列 + 预测列）
    inp_df.to_pickle(args.out_fpath)
    # 可选：保存CSV（跳过数组特征列，避免乱码）
    csv_out_path = args.out_fpath.replace(".pkl", ".csv")
    non_array_cols = [col for col in inp_df.columns if not isinstance(inp_df[col].iloc[0], np.ndarray)]
    inp_df[non_array_cols].to_csv(csv_out_path, index=False, encoding="utf-8")

    # 打印统计信息
    error_count = np.isnan(pred_act_list).sum()
    print(f"\n预测完成！结果统计：")
    print(f"- 总处理行数：{total_rows}")
    print(f"- 成功预测行数：{total_rows - error_count}")
    print(f"- 失败行数：{error_count}")
    print(f"- 结果保存路径：")
    print(f"  - Pickle（完整数据）：{args.out_fpath}")
    print(f"  - CSV（仅非数组列）：{csv_out_path}")