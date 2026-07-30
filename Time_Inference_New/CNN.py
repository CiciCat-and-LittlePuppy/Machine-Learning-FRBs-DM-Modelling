import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
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
class FRBnet(nn.Module):
    def __init__(self):
        super(FRBnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1);    self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1);   self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1);  self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1); self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1); self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3, 1, 1);self.bn6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, 3, 1, 1);self.bn7 = nn.BatchNorm2d(2048)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(7 * 7 * 2048, 16384)
        self.fc2   = nn.Linear(16384, 8192)
        self.fc3   = nn.Linear(8192, 4096)
        self.fc4   = nn.Linear(4096, 2048)
        self.fc5   = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(p=0)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        return self.fc5(x)

    def num_flat_features(self, x):
        return np.prod(x.size()[1:])

# ---------------
# DATASET
# ---------------
class FRBDataset(Dataset):
    def __init__(self, npz_file_paths, base_dir=""):
        self.file_paths = npz_file_paths
        self.directory  = base_dir
        self.transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fname     = self.file_paths[idx]
        file_path = os.path.join(self.directory, fname)
        data      = np.load(file_path, allow_pickle=True)
        image     = (
            resize_time_axis(data["wfall"])
            if "wfall" in data
            else data["dm_time"].astype(np.float32)
        )
        target = torch.tensor(float(data['params'][0]), dtype=torch.float32)
        image  = self.transform(image)
        return image, target, fname

# ---------------
# HELPERS
# ---------------
def compute_metrics(preds, targets, threshold1=1.427, errp_thresh=0.01):
    abs_diff = np.abs(preds - targets)
    errp     = abs_diff / targets
    return {
        "MAE":    np.mean(abs_diff),
        "RMSE":   np.sqrt(np.mean(abs_diff ** 2)),
        "MERRP":  np.mean(errp),
        "GoodRate": np.mean(abs_diff < threshold1),
        "GERRP":  np.mean(errp < errp_thresh),
    }

def save_predictions_txt(path, filenames, targets, preds):
    with open(path, 'w') as f:
        f.write("Filename\tTrue_DM\tPred_DM\n")
        for name, t, p in zip(filenames, targets, preds):
            f.write(f"{name}\t{t:.6f}\t{p:.6f}\n")

def plot_preds_vs_true(path, preds, targets, title="Predictions vs True"):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel("True DM")
    plt.ylabel("Predicted DM")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------------
# MAIN
# ---------------
parser = argparse.ArgumentParser(description="Validate CNN checkpoints with standardised timing")
parser.add_argument('--checkpoint_dir',      type=str, required=True)
parser.add_argument('--validation_data_dir', type=str, required=True)
parser.add_argument('--output_dir',          type=str, required=True)
parser.add_argument('--batch_size',          type=int, default=16,
                    help='Batch size for inference (default 16 — keep identical across models)')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- DataLoader ----
val_files   = [f for f in os.listdir(args.validation_data_dir) if f.endswith('.npz')]
val_dataset = FRBDataset(val_files, args.validation_data_dir)
val_loader  = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

# Initialize cumulative metrics file
all_metrics_txt = os.path.join(args.output_dir, "all_metrics.txt")
with open(all_metrics_txt, 'w') as f:
    f.write("Checkpoint\tTotalTime_s\tPerSample_ms\tMAE\tRMSE\tMERRP\tGoodRate\tGERRP\n")

for file in sorted(os.listdir(args.checkpoint_dir)):
    if not file.endswith('.pth'):
        continue

    checkpoint_path = os.path.join(args.checkpoint_dir, file)
    print(f"\nValidating {file}")

    # ---- Load model ----
    model = FRBnet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- Timed inference loop ----
    all_preds, all_targets, all_filenames = [], [], []

    inference_start = sync_and_time()

    with torch.no_grad():
        for inputs, labels, fnames in val_loader:
            inputs  = inputs.to(device)
            outputs = model(inputs).squeeze()
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_filenames.extend(fnames)

    inference_end = sync_and_time()

    # ---- Timing report ----
    n_samples     = len(all_preds)
    total_time_s  = inference_end - inference_start
    per_sample_ms = (total_time_s / n_samples) * 1000
    per_batch_ms  = (total_time_s / len(val_loader)) * 1000

    print(f"  [TIMING] Total inference : {total_time_s:.4f} s")
    print(f"  [TIMING] Per sample      : {per_sample_ms:.4f} ms  (n={n_samples})")
    print(f"  [TIMING] Per batch       : {per_batch_ms:.4f} ms  (batch_size={args.batch_size})")

    # ---- Metrics ----
    preds_arr   = np.array(all_preds)
    targets_arr = np.array(all_targets)
    metrics     = compute_metrics(preds_arr, targets_arr)
    print(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
          f"MERRP={metrics['MERRP']:.4f}  GoodRate={metrics['GoodRate']:.4f}  "
          f"GERRP={metrics['GERRP']:.4f}")

    # ---- Save predictions ----
    pred_txt = os.path.join(args.output_dir, f"{file}_predictions.txt")
    save_predictions_txt(pred_txt, all_filenames, targets_arr, preds_arr)

    # ---- Save per-checkpoint metrics ----
    metric_txt = os.path.join(args.output_dir, f"{file}_metrics.txt")
    with open(metric_txt, 'w') as mf:
        mf.write(f"Checkpoint: {file}\n")
        mf.write(f"N_samples: {n_samples}\n")
        mf.write(f"Batch_size: {args.batch_size}\n")
        mf.write(f"Device: {device}\n")
        mf.write(f"Total_inference_time_s: {total_time_s:.6f}\n")
        mf.write(f"Per_sample_ms: {per_sample_ms:.6f}\n")
        mf.write(f"Per_batch_ms: {per_batch_ms:.6f}\n")
        for k, v in metrics.items():
            mf.write(f"{k}: {v:.6f}\n")

    # ---- Scatter plot ----
    scatter_path = os.path.join(args.output_dir, f"{file}_scatter.png")
    plot_preds_vs_true(scatter_path, preds_arr, targets_arr, title=f"{file} Predictions")

    # ---- Append to cumulative summary ----
    with open(all_metrics_txt, 'a') as f:
        f.write(f"{file}\t{total_time_s:.6f}\t{per_sample_ms:.6f}\t"
                f"{metrics['MAE']:.6f}\t{metrics['RMSE']:.6f}\t{metrics['MERRP']:.6f}\t"
                f"{metrics['GoodRate']:.6f}\t{metrics['GERRP']:.6f}\n")

print(f"\nValidation complete. Summary saved to {all_metrics_txt}")
