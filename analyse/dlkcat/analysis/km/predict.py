#!/usr/bin/python
# coding: utf-8

# Author: Customized for Km fold-based prediction
# 适配需求：按每条数据的fold调用对应模型预测Km值（模型后缀.pth）

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# ====================== 1. 配置项（仅修改Km相关字段命名，路径可根据实际调整） ======================
# 预处理数据路径（Km专用）
INPUT_DIR = "/mnt/usb1/wmx/dlkcat/Data/km/input"
# 模型文件根路径（Km模型保存路径）
MODEL_DIR = "/mnt/usb1/wmx/dlkcat/Data/Results3/km/output"
# 模型文件名前缀（必须与Km训练时的命名一致）
MODEL_PREFIX = "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"
# 输出结果路径（明确标注Km）
OUTPUT_PATH = "./km_prediction.tsv"
# 超参数（必须与Km训练时完全一致，请勿修改）
RADIUS = 2
NGRAM = 3
DIM = 20
LAYER_GNN = 3
WINDOW = 11
LAYER_CNN = 3
LAYER_OUTPUT = 3


# ====================== 2. 模型定义（与Km训练代码完全一致，请勿修改） ======================
class KmPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
        super(KmPrediction, self).__init__()
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


# ====================== 3. 工具函数（增加异常处理和设备适配） ======================
def load_tensor(file_name, dtype, device):
    """加载npy文件并转换为指定类型的tensor，适配指定设备"""
    try:
        data = np.load(file_name, allow_pickle=True)
        return [dtype(d).to(device) for d in data]
    except Exception as e:
        print(f"加载文件{file_name}失败: {e}")
        sys.exit(1)


def load_pickle(file_name):
    """加载pickle字典，增加异常处理"""
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"加载pickle文件{file_name}失败: {e}")
        sys.exit(1)


class Predictor(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()  # 固定推理模式，禁用Dropout/BatchNorm

    def predict(self, data):
        """单样本预测，增加异常值处理"""
        with torch.no_grad():  # 禁用梯度计算，提升速度并节省显存
            try:
                predicted_value = self.model.forward(data)
                pred_item = predicted_value.item()
                # 处理NaN/inf异常值
                if math.isnan(pred_item) or math.isinf(pred_item):
                    return torch.tensor([0.0]).to(data[0].device)
                return predicted_value
            except Exception as e:
                print(f"预测失败: {e}")
                return torch.tensor([0.0]).to(data[0].device)


# ====================== 4. 核心预测逻辑（Km专用） ======================
if __name__ == '__main__':
    # 设备选择（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # ---------------------- 加载预处理数据（Km专用） ----------------------
    print("\n开始加载Km预处理数据...")
    # 拼接数据路径
    compounds_path = os.path.join(INPUT_DIR, "compounds.npy")
    adjacencies_path = os.path.join(INPUT_DIR, "adjacencies.npy")
    proteins_path = os.path.join(INPUT_DIR, "proteins.npy")
    regression_path = os.path.join(INPUT_DIR, "regression.npy")
    folds_path = os.path.join(INPUT_DIR, "folds.npy")
    fingerprint_dict_path = os.path.join(INPUT_DIR, "fingerprint_dict.pickle")
    word_dict_path = os.path.join(INPUT_DIR, "sequence_dict.pickle")

    # 加载核心数据
    compounds = load_tensor(compounds_path, torch.LongTensor, device)
    adjacencies = load_tensor(adjacencies_path, torch.FloatTensor, device)
    proteins = load_tensor(proteins_path, torch.LongTensor, device)
    regression = load_tensor(regression_path, torch.FloatTensor, device)  # Km的log2值
    folds = np.load(folds_path, allow_pickle=True)
    fingerprint_dict = load_pickle(fingerprint_dict_path)
    word_dict = load_pickle(word_dict_path)

    # 数据维度获取
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    # 数据校验
    data_lengths = [len(compounds), len(adjacencies), len(proteins), len(regression), len(folds)]
    if len(set(data_lengths)) != 1:
        print(f"数据样本数不匹配！各数据长度: {data_lengths}")
        sys.exit(1)
    total_samples = len(compounds)
    print(f"成功加载 {total_samples} 个Km样本")
    print(f"Fingerprint维度: {n_fingerprint}, Protein Word维度: {n_word}")

    # ---------------------- 模型缓存（避免重复加载.pth模型） ----------------------
    model_cache = {}  # key: fold, value: Predictor实例

    # ---------------------- 初始化输出文件（Km专用字段） ----------------------
    print("\n初始化Km预测结果文件...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        header = [
            "Sample_Index",          # 样本索引
            "Fold",                 # 样本所属fold
            "True_Km_log2",         # 真实Km值（log2转换后）
            "True_Km_original",     # 真实Km值（原始值，log2逆转换）
            "Pred_Km_log2",         # 预测Km值（log2转换后）
            "Pred_Km_original",     # 预测Km值（原始值，log2逆转换）
            "Model_Path"            # 使用的模型文件路径
        ]
        f.write('\t'.join(header) + '\n')

    # ---------------------- 逐样本预测（核心逻辑） ----------------------
    print("\n开始逐样本预测Km值...")
    success_count = 0
    fail_count = 0

    for idx in range(total_samples):
        # 1. 获取当前样本的fold并校验有效性
        fold = int(folds[idx])
        if fold < 0 or fold > 9:
            print(f"警告：样本{idx}的fold={fold}无效，跳过")
            fail_count += 1
            continue

        # 2. 加载/复用对应fold的Km模型（.pth）
        if fold not in model_cache:
            model_file = f"{MODEL_PREFIX}_Fold{fold}.pth"
            model_path = os.path.join(MODEL_DIR, model_file)

            if not os.path.exists(model_path):
                print(f"错误：Km模型文件{model_path}不存在，样本{idx}预测失败")
                fail_count += 1
                continue

            # 初始化模型并加载权重
            try:
                model = KmPrediction(
                    n_fingerprint=n_fingerprint,
                    n_word=n_word,
                    dim=DIM,
                    layer_gnn=LAYER_GNN,
                    window=WINDOW,
                    layer_cnn=LAYER_CNN,
                    layer_output=LAYER_OUTPUT
                ).to(device)

                # 加载.pth权重（兼容CPU/GPU）
                model.load_state_dict(torch.load(model_path, map_location=device))
                model_cache[fold] = Predictor(model)
                print(f"已加载Fold{fold}的Km模型: {model_file}")
            except Exception as e:
                print(f"加载Fold{fold}模型失败: {e}，样本{idx}预测失败")
                fail_count += 1
                continue

        # 3. 准备当前样本输入
        inputs = [
            compounds[idx],   # 化合物指纹特征
            adjacencies[idx], # 邻接矩阵
            proteins[idx]     # 蛋白质序列特征
        ]

        # 4. 执行预测
        predictor = model_cache[fold]
        pred_log2 = predictor.predict(inputs).item()
        true_log2 = regression[idx].item()

        # 5. 转换为原始Km值（log2 → 原始值）
        pred_original = math.pow(2, pred_log2)
        true_original = math.pow(2, true_log2)

        # 6. 写入结果文件
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

        success_count += 1

        # 打印进度
        if (idx + 1) % 100 == 0:
            print(f"进度：{idx + 1}/{total_samples} 样本，成功{success_count}，失败{fail_count}")

    # ---------------------- 预测完成统计 ----------------------
    print("\n===================== Km预测完成 =====================")
    print(f"总样本数: {total_samples}")
    print(f"成功预测: {success_count}")
    print(f"失败预测: {fail_count}")
    print(f"预测结果文件: {os.path.abspath(OUTPUT_PATH)}")

    # 统计各fold的预测样本数
    fold_count = defaultdict(int)
    for fold in folds:
        fold_count[int(fold)] += 1
    print("\n各Fold样本分布:")
    for fold in sorted(fold_count.keys()):
        print(f"Fold{fold}: {fold_count[fold]} 个样本")