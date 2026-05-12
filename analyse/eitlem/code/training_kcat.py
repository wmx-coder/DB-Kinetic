from torch import nn
import sys
import re
import torch
from eitlem_utils2 import Tester, Trainer, get_fold_pair_info
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
from tqdm import tqdm
from dataset2 import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse
import warnings
# 导入早停类
from early_stop import EarlyStopping

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ========== 工具函数：保存训练日志 ==========
def save_train_log(log_path, content):
    """保存单行日志到指定文件（追加模式）"""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')


# ========== 适配Fold的单任务训练函数（核心修改：新增PCC/SCC） ==========
def kineticsTrainer(kkmPath, TrainType, Type, Iteration, fold_num, log10, molType, device, dataset_root):
    # 路径新增Fold标识，避免覆盖
    train_info = f"Transfer-{TrainType}-{Type}-Fold{fold_num}-train-{Iteration}"
    result_root = f'../Result/{Type}/Fold{fold_num}'
    log_file = f'{result_root}/{train_info}/train_log.txt'  # 日志文件路径
    weight_dir = f'{result_root}/{train_info}/Weight/'  # 权重目录
    # 1. 权重文件名新增PCC/SCC占位符（和KKM保持一致）
    best_model_name = f'Eitlem_{molType}_trainR2_{{train_r2:.4f}}_devR2_{{dev_r2:.4f}}_devPCC_{{dev_pcc:.4f}}_devSCC_{{dev_scc:.4f}}_RMSE_{{rmse:.4f}}_MAE_{{mae:.4f}}.pth'

    # 若已存在训练结果，直接返回
    if os.path.exists(f'{result_root}/{train_info}'):
        return None
    # 创建目录（含日志/权重目录）
    os.makedirs(weight_dir, exist_ok=True)

    # 训练轮数（Iteration=1时改为100轮，早停生效）
    if kkmPath is not None:
        Epoch = 40 // (Iteration // 2)
    else:
        Epoch = 100

    # 2. 修复缩进错误：模型初始化不在if分支内
    if Type == 'KCAT':
        model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
    else:
        model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)

    # 迁移学习加载权重（保持原逻辑）
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        weights = model.state_dict()
        if Type == 'KCAT':
            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
        else:
            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in weights}
        weights.update(pretrained_para)
        model.load_state_dict(weights)

    """Train setting."""
    # 加载指定Fold的训练/测试集
    train_pair_info, test_pair_info = get_fold_pair_info(dataset_root, Type, fold_num)
    protein_path = "/mnt/usb1/wmx/eitlem/Data/Feature/esm2_t33_650M_UR50D_unique"
    smiles_dict_path = os.path.join(dataset_root, f"Feature/SmilesDict/{Type}_SmilesDict.pt")

    train_set = EitlemDataSet(
        train_pair_info,
        protein_path,
        smiles_dict_path,
        log10=log10,
        Type=molType
    )
    test_set = EitlemDataSet(
        test_pair_info,
        protein_path,
        smiles_dict_path,
        log10=log10,
        Type=molType
    )

    # 数据加载器
    train_loader = EitlemDataLoader(
        data=train_set,
        batch_size=200, shuffle=True, drop_last=False,
        num_workers=8, prefetch_factor=5, persistent_workers=False, pin_memory=True
    )
    valid_loader = EitlemDataLoader(
        data=test_set,
        batch_size=200, drop_last=False,
        num_workers=8, prefetch_factor=5, persistent_workers=False, pin_memory=True
    )

    model = model.to(device)

    # 优化器（保持原逻辑）
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda x: id(x) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rest_param, 'lr': 1e-4},
            {'params': model.out.parameters(), 'lr': 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)

    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)

    # 初始化早停类（patience=20，min_delta=0.001，和你的代码对齐）
    early_stopper = EarlyStopping(patience=20, min_delta=0.0001)
    best_dev_r2 = -float('inf')
    # 3. 新增PCC/SCC最优值初始化
    best_dev_pcc = -float('inf')
    best_dev_scc = -float('inf')
    best_metrics = {}  # 存储最优指标
    best_model_path = ""  # 存储最优权重完整路径

    # 写入初始日志
    init_log = f"===== Fold{fold_num} {Type} 开始训练 ====="
    print(init_log)
    save_train_log(log_file, init_log)

    # 训练循环
    for epoch in range(1, Epoch + 1):
        # 训练+验证：接收PCC/SCC返回值
        train_MAE, train_rmse, train_r2, loss_train, train_pcc, train_scc = trainer.run(
            model, train_loader, optimizer, len(train_pair_info),
            f"Fold{fold_num} {Iteration}iter epoch {epoch} train:"
        )
        MAE_dev, RMSE_dev, R2_dev, loss_dev, dev_pcc, dev_scc, y_pred_dev, y_label_dev = tester.test(
            model, valid_loader, len(test_pair_info),
            desc=f"Fold{fold_num} {Iteration}iter epoch {epoch} valid:"
        )
        scheduler.step()

        # 早停判断（按验证损失）
        is_best, stop = early_stopper.check(epoch, loss_dev)

        # 保存最优模型（新增PCC/SCC）
        if is_best:
            # 删除旧的最优权重（避免冗余）
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            # 更新最优指标（新增PCC/SCC）
            best_dev_r2 = R2_dev
            best_dev_pcc = dev_pcc
            best_dev_scc = dev_scc
            best_metrics = {
                'epoch': epoch,
                'train_MAE': train_MAE, 'train_rmse': train_rmse, 'train_r2': train_r2,
                'train_pcc': train_pcc, 'train_scc': train_scc,  # 新增训练集PCC/SCC
                'loss_train': loss_train,
                'MAE_dev': MAE_dev, 'RMSE_dev': RMSE_dev, 'R2_dev': R2_dev,
                'PCC_dev': dev_pcc, 'SCC_dev': dev_scc,  # 新增验证集PCC/SCC
                'loss_dev': loss_dev
            }
            # 生成权重文件名（填充PCC/SCC占位符）
            best_model_path = os.path.join(
                weight_dir,
                best_model_name.format(
                    train_r2=train_r2,
                    dev_r2=R2_dev,
                    dev_pcc=dev_pcc,  # 新增PCC
                    dev_scc=dev_scc,  # 新增SCC
                    rmse=RMSE_dev,
                    mae=MAE_dev
                )
            )
            # 保存最优权重
            torch.save(model.state_dict(), best_model_path)
            # 4. 日志新增PCC/SCC
            best_log = f"Epoch {epoch} | 保存最优模型 | 路径：{best_model_path} | Dev R²: {R2_dev:.4f} | Dev PCC: {dev_pcc:.4f} | Dev SCC: {dev_scc:.4f}"
            print(best_log)
            save_train_log(log_file, best_log)

        # 写入本轮日志（新增PCC/SCC）
        epoch_log = (
            f"Epoch {epoch}/{Epoch} | "
            f"Train: Loss={loss_train:.4f}, MAE={train_MAE:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}, PCC={train_pcc:.4f}, SCC={train_scc:.4f} | "
            f"Valid: Loss={loss_dev:.4f}, MAE={MAE_dev:.4f}, RMSE={RMSE_dev:.4f}, R2={R2_dev:.4f}, PCC={dev_pcc:.4f}, SCC={dev_scc:.4f}"
        )
        print(epoch_log)
        save_train_log(log_file, epoch_log)

        # 触发早停，终止训练（日志新增PCC/SCC）
        if stop:
            stop_log = f"早停触发 | 终止于Epoch {epoch} | 最优Epoch: {best_metrics['epoch']} | 最优Dev R²: {best_dev_r2:.4f} | 最优Dev PCC: {best_dev_pcc:.4f} | 最优Dev SCC: {best_dev_scc:.4f}"
            print(stop_log)
            save_train_log(log_file, stop_log)
            break

    # 加载最优模型（最终仅保留最优权重）
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        # 最终日志新增PCC/SCC
        final_log = (
            f"训练完成 | 加载最优模型（Epoch {best_metrics['epoch']}）| "
            f"最优指标：Train R²={best_metrics['train_r2']:.4f}, Dev R²={best_metrics['R2_dev']:.4f}, Dev PCC={best_metrics['PCC_dev']:.4f}, Dev SCC={best_metrics['SCC_dev']:.4f} | "
            f"最优权重路径：{best_model_path}"
        )
        print(final_log)
        save_train_log(log_file, final_log)
    pass


# ========== 适配Fold的KKM训练函数（修正变量初始化+日志） ==========
def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, fold_num, log10, molType, device, dataset_root):
    train_info = f"Transfer-{TrainType}-KKM-Fold{fold_num}-train-{Iteration}"
    result_root = f'../Result/KKM/Fold{fold_num}'
    log_file = f'{result_root}/{train_info}/train_log.txt'
    weight_dir = f'{result_root}/{train_info}/Weight/'
    # KKM最优权重新增PCC/SCC占位符
    best_model_name = f'Eitlem_{molType}_trainR2_{{train_r2:.4f}}_devR2_{{dev_r2:.4f}}_devPCC_{{dev_pcc:.4f}}_devSCC_{{dev_scc:.4f}}_RMSE_{{rmse:.4f}}_MAE_{{mae:.4f}}.pth'

    if os.path.exists(f'{result_root}/{train_info}'):
        return None
    os.makedirs(weight_dir, exist_ok=True)

    Epoch = 40  # 可根据需求改为100

    # 初始化集成模型
    model = ensemble(167, 512, 1280, 10, 0.5, 10)

    # 加载KCAT/KM权重
    kcat_pretrained = torch.load(kcatPath)
    km_pretrained = torch.load(kmPath)
    kcat_parameters = model.kcat.state_dict()
    km_parameters = model.km.state_dict()
    pretrained_kcat_para = {k: v for k, v in kcat_pretrained.items() if k in kcat_parameters}
    pretrained_km_para = {k: v for k, v in km_pretrained.items() if k in km_parameters}
    kcat_parameters.update(pretrained_kcat_para)
    km_parameters.update(pretrained_km_para)
    model.kcat.load_state_dict(kcat_parameters)
    model.km.load_state_dict(km_parameters)

    """Train setting."""
    train_pair_info, test_pair_info = get_fold_pair_info(dataset_root, 'KKM', fold_num)
    protein_path = "/mnt/usb1/wmx/eitlem/Data/Feature/esm2_t33_650M_UR50D_unique"
    smiles_dict_path = os.path.join(dataset_root, f"Feature/SmilesDict/KKM_SmilesDict.pt")

    train_set = EitlemDataSet(
        train_pair_info,
        protein_path,
        smiles_dict_path,
        log10=log10,
        Type=molType
    )
    test_set = EitlemDataSet(
        test_pair_info,
        protein_path,
        smiles_dict_path,
        log10=log10,
        Type=molType
    )

    train_loader = EitlemDataLoader(data=train_set, batch_size=16, shuffle=True, drop_last=False, num_workers=8,
                                    prefetch_factor=5, persistent_workers=False, pin_memory=True)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=16, drop_last=False, num_workers=8, prefetch_factor=5,
                                    persistent_workers=False, pin_memory=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.kcat.parameters(), 'lr': 1e-4},
        {'params': model.km.parameters(), 'lr': 1e-4},
        {'params': model.o.parameters(), 'lr': 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)

    # 初始化早停+新增PCC/SCC最优值
    early_stopper = EarlyStopping(patience=20, min_delta=0.0001)
    best_dev_r2 = -float('inf')
    best_dev_pcc = -float('inf')  # 新增初始化
    best_dev_scc = -float('inf')  # 新增初始化
    best_metrics = {}
    best_model_path = ""

    # 初始日志
    init_log = f"===== Fold{fold_num} KKM 开始训练 ====="
    print(init_log)
    save_train_log(log_file, init_log)

    # 训练循环
    for epoch in range(1, Epoch + 1):
        # 接收训练集返回值（含PCC/SCC）
        train_MAE, train_rmse, train_r2, loss_train, train_pcc, train_scc = trainer.run(
            model, train_loader, optimizer, len(train_pair_info),
            f"Fold{fold_num} {Iteration}iter epoch {epoch} train:"
        )

        # 接收验证集返回值（含PCC/SCC）
        MAE_dev, RMSE_dev, R2_dev, loss_dev, dev_pcc, dev_scc, y_pred_dev, y_label_dev = tester.test(
            model, valid_loader, len(test_pair_info),
            desc=f"Fold{fold_num} {Iteration}iter epoch {epoch} valid:"
        )

        scheduler.step()

        # 早停判断
        is_best, stop = early_stopper.check(epoch, loss_dev)

        # 保存最优模型
        if is_best:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_dev_r2 = R2_dev
            best_dev_pcc = dev_pcc
            best_dev_scc = dev_scc
            best_metrics = {
                'epoch': epoch,
                'train_MAE': train_MAE, 'train_rmse': train_rmse, 'train_r2': train_r2,
                'train_pcc': train_pcc, 'train_scc': train_scc,
                'loss_train': loss_train,
                'MAE_dev': MAE_dev, 'RMSE_dev': RMSE_dev, 'R2_dev': R2_dev,
                'PCC_dev': dev_pcc, 'SCC_dev': dev_scc,
                'loss_dev': loss_dev,
            }
            # 生成权重文件名（填充所有占位符）
            best_model_path = os.path.join(
                weight_dir,
                best_model_name.format(
                    train_r2=train_r2,
                    dev_r2=R2_dev,
                    dev_pcc=dev_pcc,
                    dev_scc=dev_scc,
                    rmse=RMSE_dev,
                    mae=MAE_dev
                )
            )
            torch.save(model.state_dict(), best_model_path)
            best_log = f"Epoch {epoch} | 保存最优模型 | Dev R²: {R2_dev:.4f} | Dev PCC: {dev_pcc:.4f} | Dev SCC: {dev_scc:.4f}"
            print(best_log)
            save_train_log(log_file, best_log)

        # 本轮日志
        epoch_log = (
            f"Epoch {epoch}/{Epoch} | "
            f"Train: Loss={loss_train:.4f}, MAE={train_MAE:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}, PCC={train_pcc:.4f}, SCC={train_scc:.4f} | "
            f"Valid: Loss={loss_dev:.4f}, MAE={MAE_dev:.4f}, RMSE={RMSE_dev:.4f}, R2={R2_dev:.4f}, PCC={dev_pcc:.4f}, SCC={dev_scc:.4f}"
        )
        print(epoch_log)
        save_train_log(log_file, epoch_log)

        if stop:
            stop_log = f"早停触发 | 最优Dev R²: {best_dev_r2:.4f} | 最优Dev PCC: {best_dev_pcc:.4f} | 最优Dev SCC: {best_dev_scc:.4f}"
            print(stop_log)
            save_train_log(log_file, stop_log)
            break

    # 加载最优模型（新增PCC/SCC日志）
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        final_log = (
            f"训练完成 | 加载最优模型（Epoch {best_metrics['epoch']}）| "
            f"最优指标：Train R²={best_metrics['train_r2']:.4f}, Dev R²={best_metrics['R2_dev']:.4f}, Dev PCC={best_metrics['PCC_dev']:.4f}, Dev SCC={best_metrics['SCC_dev']:.4f} | "
            f"最优权重路径：{best_model_path}"
        )
        print(final_log)
        save_train_log(log_file, final_log)


# ========== 新增：获取指定Fold的模型路径 ==========
def getPath(Type, TrainType, Iteration, fold_num):
    train_info = f"Transfer-{TrainType}-{Type}-Fold{fold_num}-train-{Iteration}"
    file_model = f'../Result/{Type}/Fold{fold_num}/{train_info}/Weight/'
    # 遍历权重目录，取第一个.pth文件
    fileList = [f for f in os.listdir(file_model) if f.endswith('.pth')]
    if not fileList:
        raise FileNotFoundError(f"Fold{fold_num} {Type} 未找到训练好的权重文件，路径：{file_model}")
    return os.path.join(file_model, fileList[0])


# ========== 迁移学习主函数 ==========
def TransferLearing(Iterations, TrainType, log10=False, molType='MACCSKeys', device=None,
                    dataset_root="../data/kcat/"):
    for fold_num in range(10):
        print(f"\n==================== Fold{fold_num} 训练 ====================")
        for iteration in range(1, Iterations + 1):
            if iteration == 1:
                kineticsTrainer(None, TrainType, 'KCAT', iteration, fold_num, log10, molType, device, dataset_root)
                kineticsTrainer(None, TrainType, 'KM', iteration, fold_num, log10, molType, device, dataset_root)
            else:
                kkmPath = getPath('KKM', TrainType, iteration - 1, fold_num)
                kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, fold_num, log10, molType, device, dataset_root)
                kineticsTrainer(kkmPath, TrainType, 'KM', iteration, fold_num, log10, molType, device, dataset_root)

            kcatPath = getPath('KCAT', TrainType, iteration, fold_num)
            kmPath = getPath('KM', TrainType, iteration, fold_num)
            KKMTrainer(kcatPath, kmPath, TrainType, iteration, fold_num, log10, molType, device, dataset_root)


# ========== 命令行参数解析 ==========
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True)
    parser.add_argument('-t', '--TrainType', type=str, required=True)
    parser.add_argument('-l', '--log10', type=bool, required=False, default=True)
    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys')
    parser.add_argument('-d', '--device', type=int, required=True)
    parser.add_argument('-r', '--dataset_root', type=str, required=False, default="../data/kcat/")
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"use device {device}")
    TransferLearing(
        args.Iteration,
        args.TrainType,
        args.log10,
        args.molType,
        device,
        args.dataset_root
    )