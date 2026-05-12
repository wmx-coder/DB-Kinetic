# #!/usr/bin/python
# # coding: utf-8
# # 【DMS 12 个底物 批量预测】无实验值专用 —— 直接运行！
# import os
# import torch
# import json
# import pickle
# import numpy as np
# import torch.nn as nn
# from rdkit import Chem
# from collections import defaultdict
#
# # ====================== 路径配置（你的训练字典） ======================
# TRAIN_DICT_ROOT = "/mnt/usb1/wmx/dlkcat/Data/kcat/input"
#
# # 👉 12 个 DMS 底物 JSON 文件
# JSON_FILES = [
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/EcTL_sbt1.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/EcTL_sbt2.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/HIS3_sbt1.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/HIS3_sbt2.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/HIS7_sbt1.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/HIS7_sbt2.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/si-dbs_ub_sbt1.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/si-dbs_ub_sbt2.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/Ssdata_ub_sbt1.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/Ssdata_ub_sbt2.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/Ttdata_sbt1.json",
#     "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/Ttdata_sbt2.json",
# ]
#
# MODEL_DIR = "/mnt/usb1/wmx/dlkcat/Data/Results3/kcat/output"
# MODEL_PREFIX = "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"
#
# # ====================== 超参数 ======================
# RADIUS = 2
# NGRAM = 3
# DIM = 20
# LAYER_GNN = 3
# WINDOW = 11
# LAYER_CNN = 3
# LAYER_OUTPUT = 3
#
# # ====================== 模型结构（完全不变） ======================
# class KcatPrediction(nn.Module):
#     def __init__(self, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
#         super().__init__()
#         self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
#         self.embed_word = nn.Embedding(n_word, dim)
#         self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
#         self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, 2 * window + 1, padding=window) for _ in range(layer_cnn)])
#         self.W_attention = nn.Linear(dim, dim)
#         self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
#         self.W_interaction = nn.Linear(2 * dim, 1)
#
#     def gnn(self, xs, A, layer):
#         for i in range(layer):
#             hs = torch.relu(self.W_gnn[i](xs))
#             xs = xs + torch.matmul(A, hs)
#         return torch.unsqueeze(torch.mean(xs, 0), 0)
#
#     def attention_cnn(self, x, xs, layer):
#         xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
#         for i in range(layer):
#             xs = torch.relu(self.W_cnn[i](xs))
#         xs = torch.squeeze(torch.squeeze(xs, 0), 0)
#         h = torch.relu(self.W_attention(x))
#         hs = torch.relu(self.W_attention(xs))
#         weights = torch.tanh(torch.nn.functional.linear(h, hs))
#         ys = torch.t(weights) * xs
#         return torch.unsqueeze(torch.mean(ys, 0), 0)
#
#     def forward(self, inputs):
#         fingerprints, adjacency, words = inputs
#         fp_vec = self.embed_fingerprint(fingerprints)
#         mol_vec = self.gnn(fp_vec, adjacency, LAYER_GNN)
#         word_vec = self.embed_word(words)
#         pro_vec = self.attention_cnn(mol_vec, word_vec, LAYER_CNN)
#         cat_vec = torch.cat((mol_vec, pro_vec), 1)
#         for i in range(LAYER_OUTPUT):
#             cat_vec = torch.relu(self.W_out[i](cat_vec))
#         return self.W_interaction(cat_vec)
#
# # ====================== 加载训练字典 ======================
# def load_train_dicts():
#     fp = pickle.load(open(f"{TRAIN_DICT_ROOT}/fingerprint_dict.pickle", "rb"))
#     atom = pickle.load(open(f"{TRAIN_DICT_ROOT}/atom_dict.pickle", "rb"))
#     bond = pickle.load(open(f"{TRAIN_DICT_ROOT}/bond_dict.pickle", "rb"))
#     edge = pickle.load(open(f"{TRAIN_DICT_ROOT}/edge_dict.pickle", "rb"))
#     word = pickle.load(open(f"{TRAIN_DICT_ROOT}/sequence_dict.pickle", "rb"))
#     return fp, atom, bond, edge, word
#
# # ====================== 特征提取 ======================
# def split_sequence(seq, ngram, word_dict):
#     seq = '-' + seq + '='
#     words = []
#     for i in range(len(seq) - ngram + 1):
#         key = seq[i:i+ngram]
#         words.append(word_dict.get(key, 0))
#     return np.array(words)
#
# def create_atoms(mol, atom_dict):
#     atoms = [a.GetSymbol() for a in mol.GetAtoms()]
#     for a in mol.GetAromaticAtoms():
#         atoms[a.GetIdx()] = (atoms[a.GetIdx()], "aromatic")
#     return np.array([atom_dict.get(a, 0) for a in atoms])
#
# def create_ijbond(mol, bond_dict):
#     d = defaultdict(list)
#     for b in mol.GetBonds():
#         i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
#         t = bond_dict.get(str(b.GetBondType()), 0)
#         d[i].append((j, t))
#         d[j].append((i, t))
#     return d
#
# def extract_fingerprints(atoms, bond_d, radius, fp_dict, edge_dict):
#     if len(atoms) == 1 or radius == 0:
#         return np.array([fp_dict.get(a, 0) for a in atoms])
#     nodes, edg = atoms, bond_d
#     for _ in range(radius):
#         fps = []
#         for i, je in edg.items():
#             nb = [(nodes[j], e) for j, e in je]
#             key = (nodes[i], tuple(sorted(nb)))
#             fps.append(fp_dict.get(key, 0))
#         nodes = fps
#         new_edg = defaultdict(list)
#         for i, je in edg.items():
#             for j, e in je:
#                 k = tuple(sorted((nodes[i], nodes[j])))
#                 new_edg[i].append((j, edge_dict.get((k, e), 0)))
#         edg = new_edg
#     return np.array(nodes)
#
# # ====================== 加载 10 个折模型 ======================
# def load_all_models(device, n_fp, n_word):
#     models = {}
#     print("\n正在加载 fold0 ~ fold9 模型...")
#     for fold in range(10):
#         path = f"{MODEL_DIR}/{MODEL_PREFIX}_Fold{fold}.pth"
#         model = KcatPrediction(n_fp, n_word, DIM, LAYER_GNN, WINDOW, LAYER_CNN, LAYER_OUTPUT).to(device)
#         model.load_state_dict(torch.load(path, map_location=device))
#         model.eval()
#         models[fold] = model
#         print(f"✅ Fold{fold} 加载成功")
#     return models
#
# # ====================== 预测单个 JSON 文件 ======================
# def predict_single_file(json_path, device):
#     # 输出目录 = json 所在目录，结果保存为同名 tsv
#     base_dir = os.path.dirname(json_path)
#     name = os.path.basename(json_path).replace(".json", "")
#     out_path = os.path.join(base_dir, f"{name}_pred.tsv")
#
#     print(f"\n{'='*60}")
#     print(f" 预测：{json_path}")
#     print(f"{'='*60}")
#
#     with open(json_path, "r") as f:
#         data = json.load(f)
#
#     fp_dict, atom_dict, bond_dict, edge_dict, word_dict = load_train_dicts()
#     models = load_all_models(device, len(fp_dict), len(word_dict))
#
#     with open(out_path, "w") as f:
#         f.write("index\tSmiles\tSequence\tPred_log_avg\tPred_value_avg\n")
#
#     for i, entry in enumerate(data):
#         sm = entry["Smiles"]
#         seq = entry["Sequence"]
#
#         mol = Chem.AddHs(Chem.MolFromSmiles(sm))
#         atoms = create_atoms(mol, atom_dict)
#         ij_bond = create_ijbond(mol, bond_dict)
#         fps = extract_fingerprints(atoms, ij_bond, RADIUS, fp_dict, edge_dict)
#         adj = Chem.GetAdjacencyMatrix(mol)
#         words = split_sequence(seq, NGRAM, word_dict)
#
#         fps = torch.LongTensor(fps).to(device)
#         adj = torch.FloatTensor(adj).to(device)
#         words = torch.LongTensor(words).to(device)
#
#         preds = []
#         with torch.no_grad():
#             for f in range(10):
#                 pred = models[f]([fps, adj, words]).item()
#                 preds.append(pred)
#         avg_log = np.mean(preds)
#         avg_val = 2 ** avg_log
#
#         line = f"{i}\t{sm}\t{seq}\t{avg_log:.4f}\t{avg_val:.4f}\n"
#         with open(out_path, "a") as f:
#             f.write(line)
#
#         if (i+1) % 20 == 0:
#             print(f"已完成：{i+1}/{len(data)}")
#
#     print(f"\n✅ 处理完成：{json_path}")
#     print(f"📁 输出到：{out_path}")
#
# # ====================== 批量运行 ======================
# if __name__ == "__main__":
#     device = torch.device("cuda")
#     print(f"使用设备：{device}")
#
#     for json_path in JSON_FILES:
#         if os.path.exists(json_path):
#             predict_single_file(json_path, device)
#         else:
#             print(f"⚠️ 文件不存在：{json_path}")
#
#     print("\n🎉🎉🎉 12 个 DMS 底物全部预测完成！")

#!/usr/bin/python
# coding: utf-8
# 【只预测 3 个单底物：HIS3 / HIS7 / si-dbs_ub】
import os
import torch
import json
import pickle
import numpy as np
import torch.nn as nn
from rdkit import Chem
from collections import defaultdict

# ====================== 路径配置 ======================
TRAIN_DICT_ROOT = "/mnt/usb1/wmx/dlkcat/Data/kcat/input"

# 👉 只保留这 3 个单底物 JSON
JSON_FILES = [
    "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/HIS3.json",
    "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/HIS7.json",
    "/mnt/usb1/wmx/dlkcat/dlkcat_duibi/analysis/DMS/si-dbs_ub.json",
]

MODEL_DIR = "/mnt/usb1/wmx/dlkcat/Data/Results3/kcat/output"
MODEL_PREFIX = "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"

# ====================== 超参数 ======================
RADIUS = 2
NGRAM = 3
DIM = 20
LAYER_GNN = 3
WINDOW = 11
LAYER_CNN = 3
LAYER_OUTPUT = 3

# ====================== 模型结构 ======================
class KcatPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, 2 * window + 1, padding=window) for _ in range(layer_cnn)])
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
        ys = torch.t(weights) * xs
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        fp_vec = self.embed_fingerprint(fingerprints)
        mol_vec = self.gnn(fp_vec, adjacency, LAYER_GNN)
        word_vec = self.embed_word(words)
        pro_vec = self.attention_cnn(mol_vec, word_vec, LAYER_CNN)
        cat_vec = torch.cat((mol_vec, pro_vec), 1)
        for i in range(LAYER_OUTPUT):
            cat_vec = torch.relu(self.W_out[i](cat_vec))
        return self.W_interaction(cat_vec)

# ====================== 加载字典 ======================
def load_train_dicts():
    fp = pickle.load(open(f"{TRAIN_DICT_ROOT}/fingerprint_dict.pickle", "rb"))
    atom = pickle.load(open(f"{TRAIN_DICT_ROOT}/atom_dict.pickle", "rb"))
    bond = pickle.load(open(f"{TRAIN_DICT_ROOT}/bond_dict.pickle", "rb"))
    edge = pickle.load(open(f"{TRAIN_DICT_ROOT}/edge_dict.pickle", "rb"))
    word = pickle.load(open(f"{TRAIN_DICT_ROOT}/sequence_dict.pickle", "rb"))
    return fp, atom, bond, edge, word

# ====================== 特征提取 ======================
def split_sequence(seq, ngram, word_dict):
    seq = '-' + seq + '='
    words = []
    for i in range(len(seq) - ngram + 1):
        key = seq[i:i+ngram]
        words.append(word_dict.get(key, 0))
    return np.array(words)

def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        atoms[a.GetIdx()] = (atoms[a.GetIdx()], "aromatic")
    return np.array([atom_dict.get(a, 0) for a in atoms])

def create_ijbond(mol, bond_dict):
    d = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = bond_dict.get(str(b.GetBondType()), 0)
        d[i].append((j, t))
        d[j].append((i, t))
    return d

def extract_fingerprints(atoms, bond_d, radius, fp_dict, edge_dict):
    if len(atoms) == 1 or radius == 0:
        return np.array([fp_dict.get(a, 0) for a in atoms])
    nodes, edg = atoms, bond_d
    for _ in range(radius):
        fps = []
        for i, je in edg.items():
            nb = [(nodes[j], e) for j, e in je]
            key = (nodes[i], tuple(sorted(nb)))
            fps.append(fp_dict.get(key, 0))
        nodes = fps
        new_edg = defaultdict(list)
        for i, je in edg.items():
            for j, e in je:
                k = tuple(sorted((nodes[i], nodes[j])))
                new_edg[i].append((j, edge_dict.get((k, e), 0)))
        edg = new_edg
    return np.array(nodes)

# ====================== 加载 10 折模型 ======================
def load_all_models(device, n_fp, n_word):
    models = {}
    print("\n正在加载 fold0 ~ fold9 模型...")
    for fold in range(10):
        path = f"{MODEL_DIR}/{MODEL_PREFIX}_Fold{fold}.pth"
        model = KcatPrediction(n_fp, n_word, DIM, LAYER_GNN, WINDOW, LAYER_CNN, LAYER_OUTPUT).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[fold] = model
        print(f"✅ Fold{fold} 加载成功")
    return models

# ====================== 预测 ======================
def predict_single_file(json_path, device):
    base_dir = os.path.dirname(json_path)
    name = os.path.basename(json_path).replace(".json", "")
    out_path = os.path.join(base_dir, f"{name}_pred.tsv")

    print(f"\n{'='*60}")
    print(f" 预测：{json_path}")
    print(f"{'='*60}")

    with open(json_path, "r") as f:
        data = json.load(f)

    fp_dict, atom_dict, bond_dict, edge_dict, word_dict = load_train_dicts()
    models = load_all_models(device, len(fp_dict), len(word_dict))

    with open(out_path, "w") as f:
        f.write("index\tSmiles\tSequence\tPred_log_avg\tPred_value_avg\n")

    for i, entry in enumerate(data):
        sm = entry["Smiles"]
        seq = entry["Sequence"]

        # 安全解析
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        atoms = create_atoms(mol, atom_dict)
        ij_bond = create_ijbond(mol, bond_dict)
        fps = extract_fingerprints(atoms, ij_bond, RADIUS, fp_dict, edge_dict)
        adj = Chem.GetAdjacencyMatrix(mol)
        words = split_sequence(seq, NGRAM, word_dict)

        fps = torch.LongTensor(fps).to(device)
        adj = torch.FloatTensor(adj).to(device)
        words = torch.LongTensor(words).to(device)

        preds = []
        with torch.no_grad():
            for f in range(10):
                pred = models[f]([fps, adj, words]).item()
                preds.append(pred)
        avg_log = np.mean(preds)
        avg_val = 2 ** avg_log

        line = f"{i}\t{sm}\t{seq}\t{avg_log:.4f}\t{avg_val:.4f}\n"
        with open(out_path, "a") as f:
            f.write(line)

        if (i+1) % 20 == 0:
            print(f"已完成：{i+1}/{len(data)}")

    print(f"\n✅ 完成：{out_path}")

# ====================== 主运行 ======================
if __name__ == "__main__":
    device = torch.device("cuda")

    for json_path in JSON_FILES:
        if os.path.exists(json_path):
            predict_single_file(json_path, device)
        else:
            print(f"⚠️ 不存在：{json_path}")

    print("\n🎉🎉🎉 3 个单底物预测全部完成！")