import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.signal import resample_poly

# ---------------
# TIMING UTILITIES
# ---------------
def sync_and_time():
    """Return wall-clock time after flushing any pending CUDA work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

# ---------------
# RESIZE UTILITY
# ---------------
def resize_time_axis(waterfall_data, target_size=1024):
    """
    Resize the time axis of waterfall data (frequency, time) to (frequency, target_size).
    Uses:
      - linear interpolation if time_size < target_size
      - windowed-sinc (resample_poly) if time_size > target_size
    """
    freq_size, time_size = waterfall_data.shape

    if time_size == target_size:
        return waterfall_data.astype(np.float32, copy=True)

    original_time = np.arange(time_size)
    target_time = np.linspace(0, time_size - 1, target_size)

    if time_size < target_size:
        from scipy.interpolate import interp1d
        interp_function = interp1d(
            original_time, waterfall_data, kind='linear',
            axis=1, assume_sorted=True,
        )
        resized_data = interp_function(target_time)
    else:
        resized_data = resample_poly(
            waterfall_data, up=target_size, down=time_size, axis=1,
        )

    return resized_data.astype(np.float32, copy=False)

# ---------------
# MODEL
# ---------------
class FRB_CNN_LSTM_3Conv(nn.Module):
    def __init__(self, lstm_hidden_size=128, num_lstm_layers=2):
        super(FRB_CNN_LSTM_3Conv, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
        )

        self.fc1 = nn.Linear(lstm_hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.squeeze(1)
        x = F.relu(self.bn1(self.conv1(x)));  x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)));  x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)));  x = self.pool3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------
# DATASET
# ---------------
class FRBDataset(Dataset):
    """Each sample: (1, 512, 1024) + scalar DM."""
    def __init__(self, npz_file_paths, base_dir=""):
        self.file_paths = npz_file_paths
        self.directory  = base_dir
        self.transform  = transforms.ToTensor()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename  = self.file_paths[idx]
        file_path = os.path.join(self.directory, filename)
        data      = np.load(file_path, allow_pickle=True)

        image = (
            resize_time_axis(data["wfall"])
            if "wfall" in data
            else data["dm_time"].astype(np.float32)
        )

        target = torch.tensor(float(data["params"][0]), dtype=torch.float32)
        image  = self.transform(image)
        return image, target, filename

# ---------------
# HELPERS
# ---------------
def get_npz_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.npz')]

def get_pth_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.pth')]

def compute_metrics(preds, targets, threshold1=1.084, threshold2=1.428, errp_thresh=0.01):
    abs_diff = np.abs(preds - targets)
    errp     = abs_diff / targets
    return {
        "MAE":   np.mean(abs_diff),
        "RMSE":  np.sqrt(np.mean(abs_diff ** 2)),
        "MERRP": np.mean(errp),
        "GDR_1": np.mean(abs_diff < threshold1),
        "GDR_2": np.mean(abs_diff < threshold2),
        "GERRP": np.mean(errp < errp_thresh),
    }

# ---------------
# MAIN
# ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate CNN+LSTM checkpoints with standardised timing")
    parser.add_argument('--checkpoint_dir',      type=str, required=True)
    parser.add_argument('--validation_data_dir', type=str, required=True)
    parser.add_argument('--output_dir',          type=str, required=True)
    parser.add_argument('--predict_dir',         type=str, required=True)
    parser.add_argument('--batch_size',          type=int, default=16,
                        help='Batch size for inference (default 16 — keep identical across models)')

    args = parser.parse_args()

    checkpoint_dir      = os.path.expanduser(args.checkpoint_dir)
    validation_data_dir = os.path.expanduser(args.validation_data_dir)
    output_dir          = os.path.expanduser(args.output_dir)
    predict_dir         = os.path.expanduser(args.predict_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- DataLoader ----
    val_files   = get_npz_files(validation_data_dir)
    val_dataset = FRBDataset(val_files, validation_data_dir)
    val_loader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # consistent across scripts
        num_workers=0,
    )

    plt.rcParams['font.family'] = 'Times New Roman'

    model_files = get_pth_files(checkpoint_dir)
    print(f"Found {len(model_files)} .pth file(s) in: {checkpoint_dir}")
    if not model_files:
        print("No .pth files found. Exiting.")
        exit()

    overall_results = []

    for model_fname in sorted(model_files):
        model_path = os.path.join(checkpoint_dir, model_fname)
        base_name  = os.path.splitext(model_fname)[0]
        print(f"\n--- Validating: {model_fname} ---")

        # ---- Load model (NOT timed) ----
        model = FRB_CNN_LSTM_3Conv(lstm_hidden_size=128, num_lstm_layers=2).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        )
        model.eval()

        # ---- Timed inference loop (includes data loading) ----
        all_preds, all_truths, all_fnames = [], [], []

        inference_start = sync_and_time()

        with torch.no_grad():
            for images, targets, batch_names in val_loader:
                images = images.to(device)
                outputs = model(images).squeeze()
                all_preds.extend(outputs.cpu().numpy().tolist())
                all_truths.extend(targets.numpy().tolist())
                all_fnames.extend(batch_names)

        inference_end = sync_and_time()

        # ---- Timing report ----
        n_samples        = len(all_preds)
        total_time_s     = inference_end - inference_start
        per_sample_ms    = (total_time_s / n_samples) * 1000
        n_batches_run    = len(val_loader)
        per_batch_ms     = (total_time_s / n_batches_run) * 1000

        print(f"  [TIMING] Total inference : {total_time_s:.4f} s")
        print(f"  [TIMING] Per sample      : {per_sample_ms:.4f} ms  (n={n_samples})")
        print(f"  [TIMING] Per batch       : {per_batch_ms:.4f} ms  (batch_size={args.batch_size})")

        # ---- Metrics ----
        preds_arr  = np.array(all_preds)
        truths_arr = np.array(all_truths)
        metrics    = compute_metrics(preds_arr, truths_arr)
        print(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MERRP={metrics['MERRP']:.4f}  GDR_1={metrics['GDR_1']:.4f}  "
              f"GDR_2={metrics['GDR_2']:.4f}  GERRP={metrics['GERRP']:.4f}")

        # ---- Save predictions ----
        pred_path = os.path.join(predict_dir, f"{base_name}_predictions.txt")
        with open(pred_path, 'w') as pf:
            pf.write("File_Name\tPrediction\tTruth\n")
            for name, pred, truth in zip(all_fnames, preds_arr, truths_arr):
                pf.write(f"{name}\t{pred:.6f}\t{truth:.6f}\n")

        # ---- Save metrics (including timing) ----
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
        with open(metrics_path, 'w') as mf:
            mf.write(f"Checkpoint: {model_fname}\n")
            mf.write(f"N_samples: {n_samples}\n")
            mf.write(f"Batch_size: {args.batch_size}\n")
            mf.write(f"Device: {device}\n")
            mf.write(f"Total_inference_time_s: {total_time_s:.6f}\n")
            mf.write(f"Per_sample_ms: {per_sample_ms:.6f}\n")
            mf.write(f"Per_batch_ms: {per_batch_ms:.6f}\n")
            for k, v in metrics.items():
                mf.write(f"{k}: {v:.6f}\n")

        overall_results.append((base_name, total_time_s, per_sample_ms, metrics))

        # ---- Scatter plot ----
        slope, intercept = np.polyfit(truths_arr, preds_arr, 1)
        min_val, max_val = min(truths_arr.min(), preds_arr.min()), max(truths_arr.max(), preds_arr.max())
        x_space = np.linspace(min_val, max_val, 200)

        plt.figure(figsize=(6, 6))
        plt.scatter(truths_arr, preds_arr, color='orchid', alpha=0.7, s=20, label='Data Points')
        plt.plot([min_val, max_val], [min_val, max_val], color='red',  linestyle='-',  label='Ideal: y = x')
        plt.plot(x_space, slope * x_space + intercept, color='blue', linestyle='--',
                 label=f'Regr: y={slope:.2f}x+{intercept:.2f}')
        plt.xlabel('True DM', fontsize=12)
        plt.ylabel('Predicted DM', fontsize=12)
        plt.title(f'Pred vs. True  (N={n_samples})', fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_plot.png"), dpi=300)
        plt.close()

    # ---- Overall summary ----
    overall_path = os.path.join(output_dir, "overall_performance.txt")
    with open(overall_path, 'w') as f:
        f.write("ModelName\tTotalTime_s\tPerSample_ms\tMAE\tRMSE\tMERRP\tGDR_1\tGDR_2\tGERRP\n")
        for (name, tt, ps, m) in overall_results:
            f.write(f"{name}\t{tt:.6f}\t{ps:.6f}\t"
                    f"{m['MAE']:.6f}\t{m['RMSE']:.6f}\t{m['MERRP']:.6f}\t"
                    f"{m['GDR_1']:.4f}\t{m['GDR_2']:.4f}\t{m['GERRP']:.4f}\n")

    print(f"\nAll checkpoints validated. Summary -> {overall_path}")
