import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import math


# 移除RDKit相关导入（无需解析SMILES）

class EitlemDataSet(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, SmilesDictPath, log10=False, Type='MACCSKeys'):
        super(EitlemDataSet, self).__init__()
        self.pairinfo = Pairinfo  # (esm_id, sbt_id, label)
        # 加载预生成的SMILES字典（MACCS指纹张量）
        self.smiles_dict = torch.load(SmilesDictPath)
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')  # 蛋白嵌入路径
        self.log10 = log10  # 标签归一化方式
        self.Type = Type  # 固定为MACCSKeys
        print(f"log10:{self.log10} molType:{self.Type}")

    def __getitem__(self, idx):
        pro_id = self.pairinfo[idx][0]
        smi_id = self.pairinfo[idx][1]
        value = self.pairinfo[idx][2]

        # 1. 加载蛋白质嵌入（你的软链接路径）
        protein_emb = torch.load(self.seq_path.format(pro_id))

        # 2. 加载预生成的MACCS指纹（167维）
        mol_fp = self.smiles_dict[smi_id]  # 直接获取张量，无需RDKit生成

        # 3. 标签归一化（log10/log2）
        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)

        # 4. 封装为PyG Data对象（保持原结构）
        data = Data(
            x=mol_fp.unsqueeze(0),  # 1×167
            pro_emb=protein_emb,
            value=torch.tensor(value, dtype=torch.float32)
        )
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])

    def __len__(self):
        return len(self.pairinfo)


class EitlemDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


# 保留原工具函数（shuffle/split无需修改）
def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2