import os
import torch
from tqdm import tqdm
import sys
import gc

# ========== 配置参数 ==========
# 替换为您的 ESM 特征文件根路径
ESM_ROOT = "/mnt/usb3/code/wm/esm/"
DATASET_TYPES = ['kcat', 'km', 'kcat_km']
LAYER_INDEX = 33  # Eitlem 项目使用的是第 33 层


# ==============================

def refine_esm_features(esm_root, dataset_types, layer_index):
    """
    遍历指定数据集类型下的所有 .pt 文件，提取并精简 ESM 特征。
    """
    if not os.path.exists(esm_root):
        print(f"错误：ESM 根路径不存在: {esm_root}")
        sys.exit(1)

    print("===== 开始精简 ESM 特征文件 =====")

    for type_name in dataset_types:
        print(f"\n--- 正在处理数据集: {type_name.upper()} ---")
        base_path = os.path.join(esm_root, type_name)

        if not os.path.exists(base_path):
            print(f"警告：路径 {base_path} 不存在，跳过。")
            continue

        # 查找所有 .pt 文件
        file_list = [f for f in os.listdir(base_path) if f.endswith('.pt')]

        processed_count = 0

        for filename in tqdm(file_list, desc=f"Refining {type_name}"):
            file_path = os.path.join(base_path, filename)

            try:
                # 1. 加载文件
                data = torch.load(file_path, weights_only=False, map_location="cpu")

                # 2. 检查文件是否已经被精简 (如果是 Tensor 则跳过)
                if isinstance(data, torch.Tensor):
                    # print(f"文件 {filename} 已经精简。")
                    processed_count += 1
                    continue

                    # 3. 提取第 33 层特征
                # 假设您的原始 ESM 文件结构是像 ESM 官方输出那样包含 'representations' 字典
                if 'representations' not in data or layer_index not in data['representations']:
                    print(f"\n跳过文件 {filename}: 找不到第 {layer_index} 层的特征。")
                    del data
                    gc.collect()
                    continue

                esm_feat = data['representations'][layer_index]

                # 4. 【关键步骤】切除首尾的 <CLS> 和 <EOS> Token (L+2 -> L)
                # 这一步将使其与您在 dataset2.py 中的切片逻辑保持一致
                esm_feat_refined = esm_feat[1:-1, :].float()

                # 5. 覆盖保存精简后的张量
                torch.save(esm_feat_refined, file_path)

                # 6. 清理内存
                del data, esm_feat, esm_feat_refined
                gc.collect()
                processed_count += 1

            except Exception as e:
                print(f"\n处理文件 {filename} 时发生错误: {e}")
                # 尝试清除可能残留在内存中的大对象
                gc.collect()

        print(f"数据集 {type_name.upper()} 精简完成，共处理/跳过 {processed_count} 个文件。")

    print("\n===== 所有 ESM 特征文件精简操作完成，请继续下一步。=====")
    print("现在您的模型训练速度应该会有显著提升！")


if __name__ == '__main__':
    refine_esm_features(ESM_ROOT, DATASET_TYPES, LAYER_INDEX)