#!/usr/bin/python
# coding: utf-8

# Author: Customized for fold-based prediction
# 适配需求：按每条数据的fold调用对应模型预测Kcat/Km值（模型后缀.pth）

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from collections import defaultdict

# ====================== 1. 配置项（仅修改字段命名注释，路径不变） ======================
# 预处理数据路径（kcat_km专用）
INPUT_DIR = "/mnt/usb1/wmx/dlkcat/Data/kcat_km/input"
# 模型文件根路径
MODEL_DIR = "/mnt/usb1/wmx/dlkcat/Data/Results3/kcat_km/output"
# 模型文件名前缀（保持与训练一致）
MODEL_PREFIX = "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"
# 输出结果路径（明确标注kcat_km）
OUTPUT_PATH = "./kcat_km_prediction.tsv"
# 超参数（必须与训练一致，无需修改）
RADIUS = 2
NGRAM = 3
DIM = 20
LAYER_GNN = 3
WINDOW = 11
LAYER_CNN = 3
LAYER_OUTPUT = 3


# ====================== 2. 模型定义（完全不变，与训练代码一致） ======================
class KcatPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=2 * window + 1,
            stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(torch.nn.functional.linear(h, hs))
        ys = torch.t(weights) * hs

        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, LAYER_GNN)

        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, LAYER_CNN)

        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(LAYER_OUTPUT):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction


# ====================== 3. 工具函数（仅增加异常值处理） ======================
def load_tensor(file_name, dtype):
    """加载npy文件并转换为指定类型的tensor"""
    data = np.load(file_name, allow_pickle=True)
    return [dtype(d).to(device) for d in data]


def load_pickle(file_name):
    """加载pickle字典"""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class Predictor(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()  # 推理模式

    def predict(self, data):
        """单样本预测（增加异常值处理）"""
        with torch.no_grad():  # 禁用梯度计算
            predicted_value = self.model.forward(data)
        # 处理NaN/inf异常值
        pred_item = predicted_value.item()
        if math.isnan(pred_item) or math.isinf(pred_item):
            return torch.tensor([0.0]).to(device)  # 异常值设为0
        return predicted_value


# ====================== 4. 核心预测逻辑（重点修改：字段语义+容错） ======================
if __name__ == '__main__':
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------------------- 加载预处理数据 ----------------------
    print("开始加载预处理数据（kcat_km专用）...")
    # 特征数据（kcat_km的compounds/adjacencies/proteins）
    compounds = load_tensor(os.path.join(INPUT_DIR, "compounds.npy"), torch.LongTensor)
    adjacencies = load_tensor(os.path.join(INPUT_DIR, "adjacencies.npy"), torch.FloatTensor)
    proteins = load_tensor(os.path.join(INPUT_DIR, "proteins.npy"), torch.LongTensor)
    # regression.npy：存储的是kcat/km的log2值（核心语义修正）
    regression = load_tensor(os.path.join(INPUT_DIR, "regression.npy"), torch.FloatTensor)
    # folds（匹配每条数据的fold）
    folds = np.load(os.path.join(INPUT_DIR, "folds.npy"), allow_pickle=True)
    # 字典（训练kcat_km模型时生成的字典，确保维度匹配）
    fingerprint_dict = load_pickle(os.path.join(INPUT_DIR, "fingerprint_dict.pickle"))
    word_dict = load_pickle(os.path.join(INPUT_DIR, "sequence_dict.pickle"))
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    # 数据校验
    assert len(compounds) == len(folds) == len(regression), "数据样本数不匹配！"
    total_samples = len(compounds)
    print(f"成功加载 {total_samples} 个kcat_km样本")
    print(f"Fingerprint维度: {n_fingerprint}, Word维度: {n_word}")

    # ---------------------- 模型缓存（避免重复加载.pth模型） ----------------------
    model_cache = {}  # key: fold, value: Predictor实例

    # ---------------------- 初始化输出文件（字段名修正为kcat_km） ----------------------
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        header = [
            "Sample_Index", "Fold",
            "True_kcat_km_log2", "True_kcat_km_original",  # 修正：明确是kcat/km值
            "Pred_kcat_km_log2", "Pred_kcat_km_original",  # 修正：明确是kcat/km预测值
            "Model_Path"
        ]
        f.write('\t'.join(header) + '\n')

    # ---------------------- 逐样本预测（.pth模型加载，无需修改） ----------------------
    print("开始预测kcat/km值...")
    for idx in range(total_samples):
        # 1. 获取当前样本的fold
        fold = int(folds[idx])
        if fold < 0 or fold > 9:
            print(f"警告：样本{idx}的fold={fold}无效，跳过")
            continue

        # 2. 加载/复用对应fold的.pth模型（核心：后缀保持.pth）
        if fold not in model_cache:
            # 拼接模型路径（后缀.pth，与你的模型文件一致）
            model_file = f"{MODEL_PREFIX}_Fold{fold}.pth"
            model_path = os.path.join(MODEL_DIR, model_file)

            if not os.path.exists(model_path):
                print(f"错误：模型文件{model_path}不存在，样本{idx}预测失败")
                continue

            # 初始化模型并加载.pth权重
            model = KcatPrediction(
                n_fingerprint=n_fingerprint,
                n_word=n_word,
                dim=DIM,
                layer_gnn=LAYER_GNN,
                window=WINDOW,
                layer_cnn=LAYER_CNN,
                layer_output=LAYER_OUTPUT
            ).to(device)

            # 加载.pth模型权重（无需修改，与你的保存格式一致）
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_cache[fold] = Predictor(model)
            print(f"已加载Fold{fold}模型: {model_file}")

        # 3. 准备当前样本的输入
        inputs = [
            compounds[idx],  # fingerprints
            adjacencies[idx],  # adjacency
            proteins[idx]  # words
        ]

        # 4. 预测kcat/km值
        predictor = model_cache[fold]
        pred_log2 = predictor.predict(inputs).item()
        true_log2 = regression[idx].item()

        # 5. 转换为原始kcat/km值（log2→原始，逻辑不变）
        pred_original = math.pow(2, pred_log2)
        true_original = math.pow(2, true_log2)

        # 6. 记录结果（字段名已修正为kcat_km）
        model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_Fold{fold}.pth")
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            row = [
                str(idx),
                str(fold),
                f"{true_log2:.4f}",
                f"{true_original:.4f}",
                f"{pred_log2:.4f}",
                f"{pred_original:.4f}",
                model_path
            ]
            f.write('\t'.join(row) + '\n')

        # 进度打印
        if (idx + 1) % 100 == 0:
            print(f"已完成 {idx + 1}/{total_samples} 样本预测")

    # ---------------------- 预测完成 ----------------------
    print(f"\n预测完成！kcat/km结果已保存至: {OUTPUT_PATH}")
    # 统计各fold的预测样本数
    fold_count = {}
    for fold in folds:
        fold = int(fold)
        fold_count[fold] = fold_count.get(fold, 0) + 1
    print("各Fold预测样本数：")
    for fold in sorted(fold_count.keys()):
        print(f"Fold{fold}: {fold_count[fold]} 个样本")