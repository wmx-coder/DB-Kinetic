#!/usr/bin/python
# coding: utf-8

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ====================== 1. 完全复用训练代码的模型定义 ======================
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
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, layer_cnn)

        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

# ====================== 2. 完全复用训练代码的数据加载逻辑 ======================
def load_tensor(file_name, dtype, device):
    """和训练代码完全一致的load_tensor函数"""
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

class Predictor:
    """完全对齐训练代码的预测器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, inputs):
        """
        inputs: (fingerprints, adjacency, words)
        - 所有输入均无批次维度，和训练时完全一致
        """
        pred = self.model.forward(inputs)
        return pred.squeeze().item()

# ====================== 3. 配置 + 主逻辑 ======================
if __name__ == "__main__":
    # 配置（和训练时的超参数保持一致！）
    dim = 20
    layer_gnn = 3
    window = 11
    layer_cnn = 3
    layer_output = 3
    INPUT_DIR = "/mnt/usb1/wmx/dlkcat/Data/kcat/input/"
    MODEL_DIR = "/mnt/usb1/wmx/dlkcat/Data/Results3/output/"
    MODEL_PREFIX = "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"
    OUTPUT_PATH = "./kcat_prediction.tsv"

    # 设备（和训练一致）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（完全复用训练代码的逻辑）
    print("\n开始加载预处理数据...")
    try:
        compounds = load_tensor(os.path.join(INPUT_DIR, 'compounds'), torch.LongTensor, device)
        adjacencies = load_tensor(os.path.join(INPUT_DIR, 'adjacencies'), torch.FloatTensor, device)
        proteins = load_tensor(os.path.join(INPUT_DIR, 'proteins'), torch.LongTensor, device)
        regression = load_tensor(os.path.join(INPUT_DIR, 'regression'), torch.FloatTensor, device)
        folds = np.load(os.path.join(INPUT_DIR, 'folds.npy'), allow_pickle=True)

        # 加载字典
        fingerprint_dict = load_pickle(os.path.join(INPUT_DIR, 'fingerprint_dict.pickle'))
        word_dict = load_pickle(os.path.join(INPUT_DIR, 'sequence_dict.pickle'))
        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        # 数据校验
        total_samples = len(compounds)
        assert len(compounds) == len(adjacencies) == len(proteins) == len(regression) == len(folds), \
            "数据样本数不匹配！"
        print(f"✅ 成功加载 {total_samples} 个样本")
        print(f"✅ Fingerprint字典大小: {n_fingerprint}")
        print(f"✅ Word字典大小: {n_word}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 预加载各fold的模型（缓存）
    print("\n预加载各fold模型...")
    model_cache = {}
    for fold_idx in range(10):
        model_filename = f"{MODEL_PREFIX}_Fold{fold_idx}.pth"
        model_path = os.path.join(MODEL_DIR, model_filename)
        if not os.path.exists(model_path):
            print(f"❌ Fold{fold_idx}模型文件缺失: {model_path}")
            sys.exit(1)
        # 初始化模型（超参数和训练一致）
        model = KcatPrediction(n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model_cache[fold_idx] = Predictor(model, device)
        print(f"✅ 已加载Fold{fold_idx}模型")

    # 初始化输出文件
    print("\n初始化输出文件...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        header = [
            "Sample_Index", "Fold", "True_Kcat_log2", "True_Kcat_log10",
            "True_Kcat_original", "Pred_Kcat_log2", "Pred_Kcat_log10", "Pred_Kcat_original", "Model_Path"
        ]
        f.write('\t'.join(header) + '\n')

    # 逐样本预测（完全对齐训练时的遍历逻辑）
    print("\n开始预测...")
    error_count = 0
    for sample_idx in range(total_samples):
        try:
            # 获取样本所属fold
            fold = int(folds[sample_idx])
            if fold < 0 or fold > 9:
                raise ValueError(f"无效fold值: {fold}")

            # 获取单样本输入（无批次维度，和训练一致）
            fingerprint = compounds[sample_idx]
            adjacency = adjacencies[sample_idx]
            word = proteins[sample_idx]
            true_log2 = regression[sample_idx].item()

            # 构建输入（和训练时的data[:-1]完全一致）
            inputs = (fingerprint, adjacency, word)

            # 预测
            predictor = model_cache[fold]
            pred_log2 = predictor.predict(inputs)

            # 数值转换（和训练一致：log2 → log10 → 原始值）
            true_log10 = math.log10(math.pow(2, true_log2))
            pred_log10 = math.log10(math.pow(2, pred_log2))
            true_original = math.pow(2, true_log2)
            pred_original = math.pow(2, pred_log2)

            # 保存结果
            model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_Fold{fold}.pth")
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                row = [
                    str(sample_idx), str(fold),
                    f"{true_log2:.4f}", f"{true_log10:.4f}", f"{true_original:.4f}",
                    f"{pred_log2:.4f}", f"{pred_log10:.4f}", f"{pred_original:.4f}",
                    model_path
                ]
                f.write('\t'.join(row) + '\n')

            # 进度打印
            if (sample_idx + 1) % 100 == 0:
                print(f"进度：{sample_idx + 1}/{total_samples} 样本已预测")

        except Exception as e:
            print(f"❌ 样本{sample_idx}预测失败: {e}")
            error_count += 1
            continue

    # 统计结果
    print("\n" + "="*50)
    print(f"预测完成！")
    print(f"📊 结果文件：{os.path.abspath(OUTPUT_PATH)}")
    print(f"📊 总样本数：{total_samples}")
    print(f"📊 成功预测：{total_samples - error_count}")
    print(f"📊 失败数：{error_count}")