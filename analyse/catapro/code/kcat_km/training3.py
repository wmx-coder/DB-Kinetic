import pandas as pd
import numpy as np
import torch as th
from analyse.catapro.code.km.model2 import ActivityModel, KmModel, KcatModel
from torch.utils.data import DataLoader, Dataset
from util3 import EarlyStopping, run_a_training_epoch, run_an_eval_epoch, write_logfile, \
    out_results, evaluate
from argparse import RawDescriptionHelpFormatter
import argparse
import os
import copy
import datetime


class Mydatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


class CVDataset():
    def __init__(self, fpath, batch_size=32):
        self.batch_size = batch_size

        data_df = pd.read_pickle(fpath)
        self.data_index = data_df.index.tolist()
        data_values = data_df.values
        self.data_dict = dict(zip(self.data_index, data_values[:, :-1]))

        fold_indices = data_values[:, -1]
        self.split_index_dict = {}
        for i in range(10):
            valid_indices = []
            for fold, idx in zip(fold_indices, self.data_index):
                if fold == i:
                    valid_indices.append(idx)
            train_indices = [x for x in self.data_index if not x in valid_indices]
            self.split_index_dict[i] = [train_indices, valid_indices]

    def get_dataloader(self, idx):
        train_indices, valid_indices = self.split_index_dict[idx]
        print("training number:", len(train_indices))
        print("valid number:", len(valid_indices))

        train_values = th.from_numpy(
            np.concatenate([self.data_dict[k].reshape(1, -1) for k in train_indices], axis=0).astype(np.float32))
        valid_values = th.from_numpy(
            np.concatenate([self.data_dict[k].reshape(1, -1) for k in valid_indices], axis=0).astype(np.float32))

        train_datasets = Mydatasets(train_values)
        valid_datasets = Mydatasets(valid_values)

        train_dataloader = DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True, num_workers=8)
        valid_dataloader = DataLoader(valid_datasets, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_indices, valid_dataloader


if __name__ == "__main__":
    d = "Training ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-feat_fpath", type=str, default="/mnt/usb3/code/wm/data/kcat_km/kcat_km_catapro.pkl",
                        help="Input. The path of the kcat/Km dataset.")
    parser.add_argument("-kcat_model_dpath", type=str, default="/mnt/usb/code/wm/catapro/recover/kcat_km/kcat",
                        help="Input. Directory of pretrained kcat models (0_bestmodel.pth ~ 9_bestmodel.pth).")
    parser.add_argument("-Km_model_dpath", type=str, default="/mnt/usb/code/wm/catapro/recover/kcat_km/km",
                        help="Input. Directory of pretrained Km models (0_bestmodel.pth ~ 9_bestmodel.pth).")
    parser.add_argument("-lr", type=float, default=0.01,
                        help="Input. Learning rate.")
    parser.add_argument("-rate", type=float, default=0.1,
                        help="Input. Dropout rate.")
    parser.add_argument("-alpha", type=float, default=0.5,
                        help="Input. Weight for kcat vs Km in ActivityModel.")
    parser.add_argument("-batch_size", type=int, default=16,
                        help="Input. Batch size.")
    parser.add_argument("-epochs", type=int, default=150,
                        help="Input. Epochs.")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. Device: cuda or cpu.")
    args = parser.parse_args()

    # === 固定输出目录 ===
    model_save_dir = "/mnt/usb/code/wm/catapro/recover/act_models"
    result_save_dir = "/mnt/usb/code/wm/catapro/recover/kcat_km_result"
    log_save_dir = "/mnt/usb/code/wm/catapro/recover/kcat_km_log"

    # 创建目录
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    feat_fpath = args.feat_fpath
    kcat_model_dpath = args.kcat_model_dpath
    Km_model_dpath = args.Km_model_dpath
    lr = args.lr
    rate = args.rate
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    patience = 20
    min_delta = 0.001

    cvdata = CVDataset(fpath=feat_fpath, batch_size=batch_size)
    all_pred_label = []
    all_indices = []

    for fold in range(10):
        print(f"\n=== Fold {fold} ===")
        # Load pretrained kcat and Km models
        kcatmodel = KcatModel(device=device)
        kcatmodel.load_state_dict(th.load(os.path.join(kcat_model_dpath, f"{fold}_bestmodel.pth"), map_location=device))
        kcatmodel.eval()  # Freeze if needed (optional)

        Kmmodel = KmModel(device=device)
        Kmmodel.load_state_dict(th.load(os.path.join(Km_model_dpath, f"{fold}_bestmodel.pth"), map_location=device))
        Kmmodel.eval()  # Freeze if needed (optional)

        # Create combined model
        model = ActivityModel(kcatmodel, Kmmodel, rate, alpha, device)
        model.to(device)

        train_dataloader, valid_keys, valid_dataloader = cvdata.get_dataloader(fold)
        optimizer = th.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)
        stop = EarlyStopping(patience, min_delta)
        bestmodel = None

        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record_data = []

        for epoch in range(epochs):
            print(f"Fold {fold} | Epoch: {epoch}")
            train_eval = run_a_training_epoch(model, train_dataloader, optimizer, device)
            valid_pred, valid_label, valid_eval = run_an_eval_epoch(model, valid_dataloader, device)

            record_data.append(np.concatenate([np.array([epoch]), train_eval, valid_eval], axis=0))

            # Save log to specified dir
            log_path = os.path.join(log_save_dir, f"logfile_{fold}.csv")
            write_logfile(epoch, record_data, log_path)

            is_bestmodel, stopping = stop.check(epoch, valid_eval[-1])

            if is_bestmodel:
                bestmodel = copy.deepcopy(model.state_dict())
                model_path = os.path.join(model_save_dir, f"{fold}_bestmodel.pth")
                th.save(bestmodel, model_path)

            if stopping:
                print("Early stopping!")
                break

        # end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # time_path = os.path.join(log_save_dir, f"time_running_{fold}.dat")
        # with open(time_path, 'w') as f:
        #     f.writelines('Start Time:  ' + start_time + '\n')
        #     f.writelines('End Time:  ' + end_time)

        # Reload best model and evaluate
        model = ActivityModel(KcatModel(device=device), KmModel(device=device), rate, alpha, device)
        model.load_state_dict(th.load(os.path.join(model_save_dir, f"{fold}_bestmodel.pth"), map_location=device))
        model.to(device)
        valid_pred, valid_label, valid_eval = run_an_eval_epoch(model, valid_dataloader, device)

        # Save per-fold results
        result_path = os.path.join(result_save_dir, f"results_{fold}.csv")
        out_results(valid_eval, result_path)

        all_indices += valid_keys
        all_pred_label.append(np.concatenate([valid_pred.reshape(-1, 1), valid_label.reshape(-1, 1)], axis=1))

    # Final aggregation
    all_pred_label = np.concatenate(all_pred_label, axis=0)
    final_df = pd.DataFrame(all_pred_label, index=all_indices, columns=["pred", "label"])
    final_pred_path = os.path.join(result_save_dir, "final_pred_label.csv")
    # final_df.to_csv(final_pred_path, float_format="%.4f")

    # Final metrics
    _pcc, _scc, _r2, _rmse = evaluate(all_pred_label[:, 1], all_pred_label[:, 0])
    final_results = pd.DataFrame(np.array([[_pcc, _scc, _r2, _rmse]]), columns=["PCC", "SCC", "R2", "RMSE"])
    final_results_path = os.path.join(result_save_dir, "final_results.csv")
    final_results.to_csv(final_results_path, index=False)

    print("\n✅ Training completed!")
    print(f"Results saved to: {result_save_dir}")