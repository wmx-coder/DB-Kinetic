#!/usr/bin/python
# coding: utf-8
import json
# Author: LE YUAN
# Date: 2020-10-03

import math
import pickle
import numpy as np
from collections import defaultdict
from rdkit import Chem


# 全局字典：采用defaultdict，key不存在时自动分配索引（值为当前字典长度）
word_dict = defaultdict(lambda: len(word_dict))  # 蛋白质序列ngram → 索引
atom_dict = defaultdict(lambda: len(atom_dict))  # 分子原子类型（含芳香性） → 索引
bond_dict = defaultdict(lambda: len(bond_dict))  # 分子化学键类型 → 索引
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))  # 分子指纹 → 索引
edge_dict = defaultdict(lambda: len(edge_dict))  # 分子边特征 → 索引

# 全局列表：存储最终预处理结果
proteins = list()  # 蛋白质ngram索引序列
compounds = list()  # 分子指纹索引序列
adjacencies = list()  # 分子邻接矩阵
regression = list()  # 对数转换后的kcat标签（回归任务）

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='  # 给序列添加首尾标记（区分不同位置的相同ngram）
    # 滑动窗口提取ngram，并通过word_dict转换为索引
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)  # 返回numpy数组格式

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]  # 提取所有原子的元素符号（如C、O、N）
    for a in mol.GetAromaticAtoms():  # 标记芳香性原子
        i = a.GetIdx()  # 获取原子索引
        atoms[i] = (atoms[i], 'aromatic')  # 芳香原子格式：(元素符号, 'aromatic')
    atoms = [atom_dict[a] for a in atoms]  # 原子→索引（普通原子：元素符号；芳香原子：元组）
    return np.array(atoms)  # 返回原子索引数组
def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])  # key：原子索引；value：[(相邻原子索引, 键类型索引), ...]
    for b in mol.GetBonds():  # 遍历所有化学键
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()  # 获取化学键两端的原子索引
        bond = bond_dict[str(b.GetBondType())]  # 键类型→索引（如Single、Double、Triple）
        i_jbond_dict[i].append((j, bond))  # 双向存储（i→j）
        i_jbond_dict[j].append((i, bond))  # 双向存储（j→i）
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""
    # 特殊情况：单原子分子 或 半径为0 → 指纹=原子索引本身
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms  # 初始节点特征：原子索引
        i_jedge_dict = i_jbond_dict  # 初始边特征：原子邻接+键类型
        # 迭代r次：逐步扩展子图半径（radius=2 → 提取2阶邻域信息）
        for _ in range(radius):
            # 第一步：更新节点特征（r-半径子图指纹）
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():  # 遍历每个原子
                # 收集相邻节点的特征和边特征
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                # 节点指纹：(当前节点特征, 排序后的相邻节点-边特征元组)
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])  # 指纹→索引
            nodes = fingerprints  # 更新节点特征为当前半径的指纹

            # 第二步：更新边特征（基于新的节点特征）
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    # 边特征：(两端节点特征的排序元组, 原边特征)
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict  # 更新边特征
    return np.array(fingerprints)  # 返回r-半径指纹索引数组

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)  # rdkit内置函数：生成分子邻接矩阵（0=无连接，1=有连接）
    return np.array(adjacency)  # 返回numpy数组格式

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)  # 将defaultdict转为普通dict后保存

def main() :
    with open('../../Data/database/Kcat_combination_0918.json', 'r') as infile :
        Kcat_data = json.load(infile)

    # smiles_all = [data['Smiles'] for data in Kcat_data]

    # print(len(Kcat_data))

    # smiles = "CC1=NC=C(C(=C1O)CO)CO"
    # radius = 3 # The initial setup, I suppose it is 2, but not 2.
    # 2. 设置超参数
    radius = 2  # 分子指纹的半径（WL算法迭代次数）
    ngram = 3   # 蛋白质序列的ngram长度

    """Exclude data contains '.' in the SMILES format."""
    i = 0
    for data in Kcat_data :
        smiles = data['Smiles']
        sequence = data['Sequence']
        # print(smiles)
        Kcat = data['Value']
        if "." not in smiles and float(Kcat) > 0:
            # i += 1
            # print('This is',i)
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles)) # SMILES→分子对象（添加氢原子，更完整描述分子）
            atoms = create_atoms(mol)# 提取原子索引序列
            # print(atoms)
            i_jbond_dict = create_ijbonddict(mol)# 提取原子-键字典
            # print(i_jbond_dict)

            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)# 提取r-半径指纹
            # print(fingerprints)
            compounds.append(fingerprints)

            adjacency = create_adjacency(mol)# 提取邻接矩阵
            adjacencies.append(adjacency)

            words = split_sequence(sequence,ngram)# 提取ngram索引序列
            # print(words)
            proteins.append(words)

            # print(float(Kcat))

            regression.append(np.array([math.log2(float(Kcat))]))# 5. 标签处理：kcat值→log2转换
            print(math.log2(float(Kcat)))

            # regression.append(np.array([math.log10(float(Kcat))]))
            # print(math.log10(float(Kcat)))
    # 7. 保存预处理后的张量数据（npy格式，供模型加载）
    # np.save('../../Data/kcat/input/'+'compounds', compounds,allow_pickle=True)
    # np.save('../../Data/kcat/input/'+'adjacencies', adjacencies,allow_pickle=True)
    # np.save('../../Data/kcat/input/'+'regression', regression,allow_pickle=True)
    # np.save('../../Data/kcat/input/'+'proteins', proteins,allow_pickle=True)
    # 7. 保存预处理后的张量数据（适配 NumPy >=1.24）
    output_dir = '../../Data/kcat/input/'
    np.save(output_dir + 'compounds.npy', np.array(compounds, dtype=object), allow_pickle=True)
    np.save(output_dir + 'adjacencies.npy', np.array(adjacencies, dtype=object), allow_pickle=True)
    np.save(output_dir + 'proteins.npy', np.array(proteins, dtype=object), allow_pickle=True)
    np.save(output_dir + 'regression.npy', np.array(regression, dtype=np.float32))   # 8. 保存特征映射字典
    dump_dictionary(fingerprint_dict, '../../Data/kcat/input/fingerprint_dict.pickle')
    dump_dictionary(atom_dict, '../../Data/kcat/input/atom_dict.pickle')
    dump_dictionary(bond_dict, '../../Data/kcat/input/bond_dict.pickle')
    dump_dictionary(edge_dict, '../../Data/kcat/input/edge_dict.pickle')
    dump_dictionary(word_dict, '../../Data/kcat/input/sequence_dict.pickle')


if __name__ == '__main__' :
    main()
