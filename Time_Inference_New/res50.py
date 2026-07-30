import argparse
import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
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
    target_time   = np.linspace(0, time_size - 1, target_size)

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
# DATASET
# ---------------
class FRBDataset(Dataset):
    def __init__(self, npz_file_paths, base_dir="", transform=None, mode=None):
        self.file_paths = npz_file_paths
        self.directory  = base_dir
        self.mode       = mode
        self.transform  = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1->3 channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_paths[idx])
        data      = np.load(file_path)

        image = (
            resize_time_axis(data["wfall"])
            if "wfall" in data.keys()
            else data["dm_time"]
        )

        wfall_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
        wfall_norm = (wfall_norm * 255).astype(np.uint8)
        wfall_norm = Image.fromarray(wfall_norm)

        if self.transform:
            wfall_norm = self.transform(wfall_norm)

        target = torch.tensor(float(data["params"][0]), dtype=torch.float32)
        return wfall_norm, target, self.file_paths[idx]

# ---------------
# HELPERS
# ---------------
def compute_metrics(predictions, targets, threshold_1=1.084, threshold_2=1.427,
                    error_percentage_threshold=0.01):
    errors     = predictions - targets
    abs_errors = np.abs(errors)
    errp       = abs_errors / targets
    return {
        "MAE":   np.mean(abs_errors),
        "RMSE":  np.sqrt(np.mean(errors ** 2)),
        "MEP":   np.mean(errp),
        "GER":   np.mean(errp < error_percentage_threshold),
        "GDR_1": np.mean(abs_errors < threshold_1),
        "GDR_2": np.mean(abs_errors < threshold_2),
    }

def get_npz_files(directory):
    return [
        f for f in os.listdir(directory)
        if f.endswith('.npz') and f.startswith('wfall')
    ]

# ---------------
# MAIN
# ---------------
def main():
    parser = argparse.ArgumentParser(description="Validate ResNet50 on 6-fold best models with standardised timing")
    parser.add_argument('--val_data_dir',   type=str, required=True)
    parser.add_argument('--folds_base_dir', type=str, required=True)
    parser.add_argument('--threshold_1',    type=float, default=1.084)
    parser.add_argument('--threshold_2',    type=float, default=1.43)
    parser.add_argument('--batch_size',     type=int,   default=16,
                        help='Batch size for inference (default 16 — keep identical across models)')

    args = parser.parse_args()

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_files = get_npz_files(args.val_data_dir)
    if not val_files:
        print(f"No NPZ files found in {args.val_data_dir}. Exiting.")
        return

    # ---- DataLoader ----
    val_dataset = FRBDataset(val_files, base_dir=args.val_data_dir, mode='test')
    val_loader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # consistent across scripts
        num_workers=0,
    )

    for fold_idx in range(1, 7):
        print(f"\n=== Validating fold_{fold_idx} ===")

        fold_sub_dir       = os.path.join(args.folds_base_dir, f"fold_{fold_idx}")
        best_mae_model_path = os.path.join(fold_sub_dir, "best_mae.pth")

        if not os.path.isfile(best_mae_model_path):
            print(f"  WARNING: No best_mae.pth found in {fold_sub_dir}, skipping.")
            continue

        # ---- Load model ----
        model = resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc.requires_grad = True

        model = model.to(device)
        checkpoint = torch.load(best_mae_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # ---- Timed inference loop ----
        all_preds, all_targets, all_filenames = [], [], []

        inference_start = sync_and_time()

        with torch.no_grad():
            for batch_images, batch_targets, batch_names in val_loader:
                batch_images = batch_images.to(device)
                outputs      = model(batch_images)
                all_preds.append(outputs.squeeze().cpu().numpy())
                all_targets.append(batch_targets.squeeze().numpy())
                all_filenames.extend(batch_names)

        inference_end = sync_and_time()

        # ---- Timing report ----
        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        n_samples     = len(all_preds)
        total_time_s  = inference_end - inference_start
        per_sample_ms = (total_time_s / n_samples) * 1000
        per_batch_ms  = (total_time_s / len(val_loader)) * 1000

        print(f"  [TIMING] Total inference : {total_time_s:.4f} s")
        print(f"  [TIMING] Per sample      : {per_sample_ms:.4f} ms  (n={n_samples})")
        print(f"  [TIMING] Per batch       : {per_batch_ms:.4f} ms  (batch_size={args.batch_size})")

        # ---- Metrics ----
        metrics = compute_metrics(all_preds, all_targets,
                                  threshold_1=args.threshold_1,
                                  threshold_2=args.threshold_2)
        print(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MEP={metrics['MEP']:.4f}  GER={metrics['GER']:.4f}  "
              f"GDR_1={metrics['GDR_1']:.4f}  GDR_2={metrics['GDR_2']:.4f}")

        # ---- Save validation results (including timing) ----
        result_txt_path = os.path.join(fold_sub_dir, "validation_results.txt")
        with open(result_txt_path, 'w') as f:
            f.write(f"Validation Metrics on New Data\n")
            f.write(f"Model checkpoint: best_mae.pth\n")
            f.write(f"N_samples: {n_samples}\n")
            f.write(f"Batch_size: {args.batch_size}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Total_inference_time_s: {total_time_s:.6f}\n")
            f.write(f"Per_sample_ms: {per_sample_ms:.6f}\n")
            f.write(f"Per_batch_ms: {per_batch_ms:.6f}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")

        # ---- Save per-sample predictions ----
        dm_pred_txt_path = os.path.join(fold_sub_dir, "dm_predictions.txt")
        with open(dm_pred_txt_path, 'w') as f:
            f.write("filename\ttrue_dm\tpredicted_dm\n")
            for fname, true_val, pred_val in zip(all_filenames, all_targets, all_preds):
                f.write(f"{fname}\t{true_val:.6f}\t{pred_val:.6f}\n")

        # ---- Scatter plot ----
        slope, intercept = np.polyfit(all_targets, all_preds, 1)
        xvals = np.linspace(min(all_targets), max(all_targets), 100)

        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets, all_preds, color='pink', alpha=0.5, label='Data Points')
        plt.plot(xvals, xvals, '-', color='red', label='Ideal: y = x')
        plt.plot(xvals, slope * xvals + intercept, '--', color='blue',
                 label=f"Regression: y={slope:.3f}x+{intercept:.3f}")
        plt.title(f"Fold {fold_idx} (n={n_samples})\n"
                  f"Regression: y = {slope:.3f}x + {intercept:.3f}")
        plt.xlabel("True DM")
        plt.ylabel("Predicted DM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fold_sub_dir, "dm_scatter_plot.png"), dpi=150)
        plt.close()

    print("\nValidation complete for all folds.")


if __name__ == "__main__":
    main()
