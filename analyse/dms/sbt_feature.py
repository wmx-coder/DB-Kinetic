# import numpy as np
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import MACCSkeys
# import torch
# from transformers import T5EncoderModel, T5Tokenizer
# import os
# import gc
#
# # ====================== ✅ 你原来的函数 —— 完全不动 ======================
# def GetMACCSKeys(smiles_list):
#     """Output: np.array, size is 167."""
#     N_smiles = len(smiles_list)
#     if len(set(smiles_list)) == 1:
#         mol = Chem.MolFromSmiles(smiles_list[0])
#         fp = MACCSkeys.GenMACCSKeys(mol)
#         fp_str = fp.ToBitString()
#         fp_array = np.array([int(i) for i in fp_str])
#         final_values = np.concatenate([fp_array.reshape(1, -1)] * N_smiles, axis=0)
#     else:
#         final_values = []
#         for smile in smiles_list:
#             mol = Chem.MolFromSmiles(smile)
#             fp = MACCSkeys.GenMACCSKeys(mol)
#             fp_str = fp.ToBitString()
#             fp_array = np.array([int(i) for i in fp_str])
#             final_values.append(fp_array.reshape(1, -1))
#         final_values = np.concatenate(final_values, axis=0)
#     return final_values
#
# def get_molT5_embed(smiles_list, Molt5_model):
#     tokenizer = T5Tokenizer.from_pretrained(Molt5_model)
#     model = T5EncoderModel.from_pretrained(Molt5_model)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
#     N_smiles = len(smiles_list)
#     if len(set(smiles_list)) == 1:
#         input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids.to(device)
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids)
#         last_hidden_states = outputs.last_hidden_state.cpu()
#         embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().numpy()
#         final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
#     else:
#         final_values = []
#         for smile in smiles_list:
#             input_ids = tokenizer(smile, return_tensors="pt").input_ids.to(device)
#             with torch.no_grad():
#                 outputs = model(input_ids=input_ids)
#             last_hidden_states = outputs.last_hidden_state.cpu()
#             embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().numpy()
#             final_values.append(embed.reshape(1, -1))
#         final_values = np.concatenate(final_values, axis=0)
#     return final_values
#
# # ====================== ✅ 只加：底物拆分 + 分别生成 feat1 / feat2 ======================
# def process_split_substrate(df, molt5_model_path):
#     smi_list = df["reactant_smiles"].tolist()
#
#     smi1_list = []
#     smi2_list = []
#
#     # 1. 拆分 A.B → A 和 B
#     for smi in smi_list:
#         if "." in smi:
#             a, b = smi.split(".", 1)
#         else:
#             a = smi
#             b = a  # 没有第二个底物就复制自己（保证维度一致）
#
#         smi1_list.append(a)
#         smi2_list.append(b)
#
#     # 2. 用你原版函数计算特征
#     print("  处理底物1...")
#     m1 = get_molT5_embed(smi1_list, molt5_model_path)
#     c1 = GetMACCSKeys(smi1_list)
#     f1 = np.concatenate([m1, c1], axis=1)
#
#     print("  处理底物2...")
#     m2 = get_molT5_embed(smi2_list, molt5_model_path)
#     c2 = GetMACCSKeys(smi2_list)
#     f2 = np.concatenate([m2, c2], axis=1)
#
#     # 3. 放入 dataframe
#     if "sbt_feat" in df.columns:
#         df.drop(columns=["sbt_feat"], inplace=True)
#
#     df["sbt_feat1"] = [x for x in f1]
#     df["sbt_feat2"] = [x for x in f2]
#
#     return df
#
# # ====================== ✅ 批量处理你的文件 ======================
# if __name__ == "__main__":
#     MOLT5_MODEL = "/mnt/usb1/wmx/molt5"
#
#     pkl_files = [
#         "/mnt/usb1/wmx/catapro/analyse/dms/EcTL_with_sub_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/Ssdata_ub_feats.pkl",
#         "/mnt/usb1/wmx/catapro/analyse/dms/Ttdata_with_sub_feats.pkl",
#         # "/mnt/usb1/wmx/catapro/analyse/dms/Tmdata_with_sub_feats.pkl"
#     ]
#
#     for fp in pkl_files:
#         print(f"\n处理：{fp}")
#         df = pd.read_pickle(fp)
#         df = process_split_substrate(df, MOLT5_MODEL)
#         df.to_pickle(fp)
#         print(f"✅ 完成！已生成 sbt_feat1 / sbt_feat2")
#
#     print("\n🎉 全部文件处理完毕！")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import torch
from transformers import T5EncoderModel, T5Tokenizer
import os
import gc

# ====================== ✅ 你原来的函数 —— 完全不动 ======================
def GetMACCSKeys(smiles_list):
    """Output: np.array, size is 167."""
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        mol = Chem.MolFromSmiles(smiles_list[0])
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_str = fp.ToBitString()
        fp_array = np.array([int(i) for i in fp_str])
        final_values = np.concatenate([fp_array.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            mol = Chem.MolFromSmiles(smile)
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_str = fp.ToBitString()
            fp_array = np.array([int(i) for i in fp_str])
            final_values.append(fp_array.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values

def get_molT5_embed(smiles_list, Molt5_model):
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model)
    model = T5EncoderModel.from_pretrained(Molt5_model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state.cpu()
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            input_ids = tokenizer(smile, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state.cpu()
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().numpy()
            final_values.append(embed.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values

# ====================== ✅ 单底物：只生成一套特征 ======================
def process_single_substrate(df, molt5_model_path):
    smi_list = df["reactant_smiles"].tolist()

    print("  单底物：只生成一套底物特征...")
    m = get_molT5_embed(smi_list, molt5_model_path)
    c = GetMACCSKeys(smi_list)
    feat = np.concatenate([m, c], axis=1)

    # 只保留 sbt_feat，不拆分 1 和 2
    if "sbt_feat1" in df.columns:
        df.drop(columns=["sbt_feat1", "sbt_feat2"], inplace=True)
    df["sbt_feat"] = [x for x in feat]
    return df

# ====================== ✅ 只处理这 3 个单底物！======================
if __name__ == "__main__":
    MOLT5_MODEL = "/mnt/usb1/wmx/molt5"

    # 👉 只运行这 3 个单底物文件
    pkl_files = [
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS3_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/HIS7_feats.pkl",
        "/mnt/usb1/wmx/catapro/analyse/dms/si-dbs_ub_feats.pkl",
    ]

    for fp in pkl_files:
        print(f"\n处理单底物：{fp}")
        df = pd.read_pickle(fp)
        df = process_single_substrate(df, MOLT5_MODEL)
        df.to_pickle(fp)
        print(f"✅ 完成！只保留单底物特征 sbt_feat")

    print("\n🎉 3 个单底物文件处理完毕！不再分 sbt1 / sbt2！")