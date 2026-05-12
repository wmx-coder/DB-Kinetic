#!/usr/bin/python
# coding: utf-8

import pickle
import sys
import timeit
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


# ===================== 模型定义 =====================
class KcatPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)

        # nn.init.normal_(self.embed_fingerprint.weight, std=0.1)
        # nn.init.normal_(self.embed_word.weight, std=0.1)

        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, 2 * window + 1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = F.leaky_relu(self.W_gnn[i](xs), 0.1)
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """添加内部监控的 CNN 模块"""
        # 监控 1: Embedding 出来的原始值
        if not hasattr(self, 'diag_done'):
            print(f"    [Diag] Protein Embed Output Std: {xs.std().item():.8f}")

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = F.leaky_relu(self.W_cnn[i](xs), 0.1)
            # 监控 2: 卷积后的值
            if not hasattr(self, 'diag_done'):
                print(f"    [Diag] After CNN Layer {i} Std: {xs.std().item():.8f}")

        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = F.leaky_relu(self.W_attention(x), 0.1)
        hs = F.leaky_relu(self.W_attention(xs), 0.1)

        # 监控 3: Attention 权重
        weights = torch.tanh(F.linear(h, hs))
        if not hasattr(self, 'diag_done'):
            print(f"    [Diag] Attention Weights Max: {weights.max().item():.8f}")

        ys = torch.t(weights) * hs
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        compound_vector = self.gnn(self.embed_fingerprint(fingerprints), adjacency, len(self.W_gnn))
        protein_vector = self.attention_cnn(compound_vector, self.embed_word(words), len(self.W_cnn))

        cat_vector = torch.cat((compound_vector, protein_vector), 1)

        # 监控 4: 拼接后的向量
        if not hasattr(self, 'diag_done'):
            print(f"    [Diag] Cat Vector (Protein Part) Mean: {cat_vector[0, dim:].mean().item():.8f}")

        for j in range(len(self.W_out)):
            cat_vector = F.leaky_relu(self.W_out[j](cat_vector), 0.1)

        interaction = self.W_interaction(cat_vector)
        self.diag_done = True  # 只打印一次
        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        correct_interaction = correct_interaction.view(1, 1)
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            # 这里的 backward 是关键，我们检查它是否报错
            return loss, correct_interaction.item(), predicted_interaction.item()
        else:
            return correct_interaction.item(), predicted_interaction.item()

# ===================== 训练与测试逻辑 =====================
class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        trainCorrect, trainPredict = [], []

        for data in dataset:
            loss, correct_val, predict_val = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

            # log2 -> log10 转换
            trainCorrect.append(math.log10(math.pow(2, correct_val)))
            trainPredict.append(math.log10(math.pow(2, predict_val)))

        rmse_train = np.sqrt(mean_squared_error(trainCorrect, trainPredict))
        r2_train = r2_score(trainCorrect, trainPredict)
        return loss_total, rmse_train, r2_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        testY, testPredict = [], []
        for data in dataset:
            correct_val, predict_val = self.model(data, train=False)
            testY.append(math.log10(math.pow(2, correct_val)))
            testPredict.append(math.log10(math.pow(2, predict_val)))

        MAE = np.mean(np.abs(np.array(testY) - np.array(testPredict)))
        RMSE = np.sqrt(mean_squared_error(testY, testPredict))
        R2 = r2_score(testY, testPredict)
        PCC, _ = pearsonr(testY, testPredict)
        SCC, _ = spearmanr(testY, testPredict)
        return MAE, RMSE, R2, np.nan_to_num(PCC), np.nan_to_num(SCC)


# ===================== 辅助工具 =====================
def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def print_epoch_params_stats(model, fold_idx, epoch, lr_current):
    embed_word_weight = model.embed_word.weight.detach().cpu().numpy()
    ew_std = np.std(embed_word_weight)
    ew_mean = np.mean(embed_word_weight)
    print(
        f"  [Fold {fold_idx} Epoch {epoch}] lr={lr_current:.6f} | Protein_Embed: mean={ew_mean:.6f}, std={ew_std:.8f}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    args = sys.argv[1:]
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration, setting) = args

    dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration = map(int,
                                                                                     [dim, layer_gnn, window, layer_cnn,
                                                                                      layer_output, decay_interval,
                                                                                      iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir_input = '../../Data/kcat/input/'
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor, device)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor, device)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor, device)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor, device)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    folds = np.load(dir_input + 'folds.npy', allow_pickle=True)

    dataset = list(zip(compounds, adjacencies, proteins, interactions))

    output_dir = '../../Data/Results3/output/'
    os.makedirs(output_dir, exist_ok=True)
    file_MAEs_total = os.path.join(output_dir, f"10fold_MAEs--{setting}.txt")

    with open(file_MAEs_total, 'w') as f:
        f.write('Fold\tEpoch\tTime\tRMSE_tr\tR2_tr\tMAE_te\tRMSE_te\tR2_te\tPCC_te\tSCC_te\n')

    for fold_idx in range(10):
        print(f"\n### Starting Fold {fold_idx} ###")
        test_mask = (folds == fold_idx)
        dataset_train = [dataset[i] for i in range(len(dataset)) if not test_mask[i]]
        dataset_test = [dataset[i] for i in range(len(dataset)) if test_mask[i]]

        torch.manual_seed(1234)
        model = KcatPrediction(len(fingerprint_dict), len(word_dict), dim, layer_gnn, window, layer_cnn,
                               layer_output).to(device)
        trainer = Trainer(model, lr, weight_decay)
        tester = Tester(model)

        current_lr = lr
        start_fold = timeit.default_timer()

        for epoch in range(1, iteration + 1):
            if epoch % decay_interval == 0:
                current_lr *= lr_decay
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = current_lr

            loss_train, rmse_train, r2_train = trainer.train(dataset_train)
            MAE_test, RMSE_test, R2_test, PCC_test, SCC_test = tester.test(dataset_test)

            time_epoch = timeit.default_timer() - start_fold
            res = [fold_idx, epoch, round(time_epoch, 1), round(rmse_train, 4), round(r2_train, 4),
                   round(MAE_test, 4), round(RMSE_test, 4), round(R2_test, 4), round(PCC_test, 4), round(SCC_test, 4)]

            print("\t".join(map(str, res)))
            print_epoch_params_stats(model, fold_idx, epoch, current_lr)

            with open(file_MAEs_total, 'a') as f:
                f.write("\t".join(map(str, res)) + "\n")

        torch.save(model.state_dict(), os.path.join(output_dir, f"{setting}_Fold{fold_idx}.pth"))

    print(f"\nTask Finished! Results saved to: {output_dir}")