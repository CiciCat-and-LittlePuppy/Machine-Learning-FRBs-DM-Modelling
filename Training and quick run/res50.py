import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import os
import time
from collections import defaultdict
from torchvision.models import resnet50
from PIL import Image
import random

start_time = time.time()

random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

if 'SLURM_CPUS_PER_TASK' in os.environ:
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    print("number of cpus:", int(os.environ['SLURM_CPUS_PER_TASK']))

def get_npz_files(directory):
    npz_files = [
        file for file in os.listdir(directory)
        if file.endswith('.npz') and file.startswith('wfall')
    ]
    return npz_files

# ------------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='FRB Net')
parser.add_argument('--train_dir', type=str, required=True,
                    help='Directory containing the training NPZ files')
parser.add_argument('--test_dir', type=str, required=True,
                    help='Directory containing the testing NPZ files')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory for the outputs (checkpoints & results)')
parser.add_argument('--plot_dir', type=str, required=True,
                    help='Directory for the final plot')
parser.add_argument('--train_pred_dir', type=str, required=True,
                    help='Directory for training prediction')
args = parser.parse_args()

train_dir = os.path.expanduser(args.train_dir)
test_dir = os.path.expanduser(args.test_dir)
output_dir = os.path.expanduser(args.output_dir)
plot_dir = os.path.expanduser(args.plot_dir)
train_pred_dir = os.path.expanduser(args.train_pred_dir)

# ------------------------------------------------------------------
# Gather file lists for training & testing
# ------------------------------------------------------------------
train_paths = get_npz_files(train_dir)
test_paths = get_npz_files(test_dir)

print(f"Found {len(train_paths)} training files in {train_dir}")
print(f"Found {len(test_paths)} testing files in {test_dir}")

# Make sure the results directory exists
res_dir = os.path.join(output_dir, "res")
os.makedirs(res_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(train_pred_dir, exist_ok=True)

# ------------------------------------------------------------------
# Dataset Definition
# ------------------------------------------------------------------
class FRBDataset(Dataset):
    def __init__(self, npz_file_paths, base_dir="", transform=None, mode="train"):
        self.file_paths = npz_file_paths
        self.directory = base_dir
        self.mode = mode
        # If no transform is provided, use a default transform for ResNet
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet input size
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # convert 1 channel => 3 channels
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_paths[idx])
        data = np.load(file_path)

        if "wfall" in data.keys():
            image = data["wfall"]
        else:
            image = data["dm_time"]

        if self.mode == 'train':
            noise_background = np.random.rand(*image.shape)
            wfall_with_noise = image + noise_background
        else:
            wfall_with_noise = image

        # Normalization
        wfall_with_noise = (
            wfall_with_noise - np.min(wfall_with_noise)
        ) / (np.max(wfall_with_noise) - np.min(wfall_with_noise))

        wfall_with_noise = (wfall_with_noise * 255).astype(np.uint8)
        wfall_with_noise = Image.fromarray(wfall_with_noise)

        # Apply transforms
        if self.transform:
            wfall_with_noise = self.transform(wfall_with_noise)

        target = np.array(float(data['params'][0]), dtype=np.float32)
        target = torch.tensor(target)

        return wfall_with_noise, target, self.file_paths[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(random_seed)

# ------------------------------------------------------------------
# Create Datasets & Loaders
# ------------------------------------------------------------------
train_dataset = FRBDataset(train_paths, base_dir=train_dir, mode='train')
test_dataset = FRBDataset(test_paths, base_dir=test_dir, mode='test')

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g
)

# ------------------------------------------------------------------
# Model Definition (ResNet-50)
# ------------------------------------------------------------------
model = resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze some layers
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
model.fc.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------------------------------------------------
# Loss & Optimizer
# ------------------------------------------------------------------
loss_fn = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

# ------------------------------------------------------------------
# Training Parameters
# ------------------------------------------------------------------
num_epochs = 150
train_losses = []
test_losses = []

# For storing raw predictions
train_vals = defaultdict(list)
train_vals_idx = defaultdict(list)
test_vals = defaultdict(list)
test_vals_idx = defaultdict(list)

# ------------------------------------------------------------------
# Utility functions for metrics
# ------------------------------------------------------------------
def compute_metrics(predictions, targets, error_thresholds=(1.427, 1.084), error_percentage_threshold = 0.01):
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    error_percentages = np.abs(errors) / targets
    mean_error_percentage = np.mean(error_percentages)
    good_rates = []
    for t in error_thresholds:
        fraction_good = np.mean(np.abs(errors) < t)
        good_rates.append(fraction_good)
    good_errp_rate = np.mean(error_percentages < error_percentage_threshold)
    return mae, rmse, mean_error_percentage, good_errp_rate, good_rates

# ------------------------------------------------------------------
# Prepare output file
# ------------------------------------------------------------------
output_txt_path = os.path.join(res_dir, "output.txt")

if not os.path.exists(output_txt_path) or os.path.getsize(output_txt_path) == 0:
    with open(output_txt_path, 'w') as file:
        file.write(
            "Epoch "
            "Train_Loss "
            "Test_Loss "
            "Train_MAE "
            "Test_MAE "
            "Train_RMSE "
            "Test_RMSE "
            "Train_Merrp "
            "Test_Merrp "
            "Train_good_ERRP_Rate "
            "Test_good_ERRP_Rate "
            "Train_GoodRate_1.427 "
            "Train_GoodRate_1.084 "
            "Test_GoodRate_1.427 "
            "Test_GoodRate_1.084\n"
        )

# ------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------
for epoch in range(num_epochs):
    train_pred_file_path = os.path.join(train_pred_dir, f"{epoch}_train_preds.txt")
    with open(train_pred_file_path, 'w') as tpf:
        tpf.write("Filenames\tTrue_DM\tPred_DM\n")
    test_pred_file_path = os.path.join(train_pred_dir, f"{epoch}_test_preds.txt")
    with open(test_pred_file_path, 'w') as vpf:
        vpf.write("filename\tTrue_DM\tPred_DM\n")

    # ===========================
    # TRAINING
    # ===========================
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_tgts = []
    all_train_filenames = []

    for i, (inputs, targets, idx) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        all_train_preds.append(outputs.detach().cpu().numpy())
        all_train_tgts.append(targets.detach().cpu().numpy())
        all_train_filenames.extend(idx)
        train_vals[epoch] += torch.stack((outputs.detach().cpu().squeeze(),
                                          targets.detach().cpu().squeeze()), dim=1).numpy().tolist()
        train_vals_idx[epoch] += idx

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Compute training metrics
    all_train_preds = np.concatenate(all_train_preds).ravel()
    all_train_tgts = np.concatenate(all_train_tgts).ravel()
    train_mae, train_rmse, train_mep, train_good_errp_rate, train_good_rates = compute_metrics(all_train_preds, all_train_tgts)

    with open(train_pred_file_path, 'a') as tpf:
        for name, tval, pval in zip(all_train_filenames, all_train_tgts, all_train_preds):
            tpf.write(f"{name}\t{tval:.8f}\t{pval:.8f}\n")

    # ===========================
    # VALIDATION / TESTING
    # ===========================
    model.eval()
    running_loss = 0.0
    all_test_preds = []
    all_test_tgts = []
    all_test_filenames = []

    with torch.no_grad():
        for inputs, targets, idx in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            all_test_preds.append(outputs.detach().cpu().numpy())
            all_test_tgts.append(targets.detach().cpu().numpy())
            all_test_filenames.extend(idx)
            test_vals[epoch] += torch.stack((outputs.detach().cpu().squeeze(),
                                             targets.detach().cpu().squeeze()), dim=1).numpy().tolist()
            test_vals_idx[epoch] += idx

    epoch_test_loss = running_loss / len(test_loader.dataset)
    test_losses.append(epoch_test_loss)

    # Compute testing metrics
    all_test_preds = np.concatenate(all_test_preds).ravel()
    all_test_tgts = np.concatenate(all_test_tgts).ravel()
    test_mae, test_rmse, test_mep, test_good_errp_rate, test_good_rates = compute_metrics(all_test_preds, all_test_tgts)

    with open(test_pred_file_path, 'a') as tpf:
        for name, tval, pval in zip(all_test_filenames, all_test_tgts, all_test_preds):
            tpf.write(f"{name}\t{tval:.8f}\t{pval:.8f}\n")

    # ===========================
    # Print to console
    # ===========================
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")
    print(f"  Train MAE:  {train_mae:.4f}       | Test MAE:  {test_mae:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}      | Test RMSE: {test_rmse:.4f}")
    print(f"  Train MEP: {train_mep:.4f}        | Test MEP: {test_mep:.4f}")
    print(f"  Train Good ERRP rate: {train_good_errp_rate:.4f} | Test Good ERRP rate: {test_good_errp_rate:.4f}")
    print(f"  Train Good Rate (<1.427): {train_good_rates[0]:.4f}, (<1.084): {train_good_rates[1]:.4f}")
    print(f"  Test  Good Rate (<1.427): {test_good_rates[0]:.4f}, (<1.084): {test_good_rates[1]:.4f}")

    # ===========================
    # Save checkpoint (optional)
    # ===========================
    if epoch >= 15:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }
        checkpoint_path = os.path.join(output_dir, f'resnet_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

    # ===========================
    # Write row to output file
    # ===========================
    with open(output_txt_path, 'a') as file:
        file.write(
            f"{epoch+1} "               # Epoch
            f"{epoch_train_loss:.6f} "  # Train_Loss
            f"{epoch_test_loss:.6f} "   # Test_Loss
            f"{train_mae:.6f} "         # Train_MAE
            f"{test_mae:.6f} "          # Test_MAE
            f"{train_rmse:.6f} "        # Train_RMSE
            f"{test_rmse:.6f} "         # Test_RMSE
            f"{train_mep:.6f} "
            f"{test_mep:.6f} "
            f"{train_good_errp_rate:.6f} "
            f"{test_good_errp_rate:.6f} "
            f"{train_good_rates[0]:.6f} "  # Train_GoodRate_1.427
            f"{train_good_rates[1]:.6f} "  # Train_GoodRate_1.084
            f"{test_good_rates[0]:.6f} "   # Test_GoodRate_1.427
            f"{test_good_rates[1]:.6f}\n"  # Test_GoodRate_1.084
        )

# ------------------------------------------------------------------
# Save final arrays of predictions (optional)
# ------------------------------------------------------------------
data_res = {
    "train": train_vals,
    "test": test_vals,
    "train_idx": train_vals_idx,
    "test_idx": test_vals_idx
}
out_path = os.path.join(res_dir, "resnetdt_1.npz")
np.savez(out_path, data_res)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and Testing Error per Epoch')
plt.yscale('log')
plt.tick_params(axis='y', which='both', labelsize=8)
plt.legend()
plot_path = os.path.join(plot_dir, "learning_curve.png")
plt.savefig(plot_path, dpi=150)
plt.close()

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
