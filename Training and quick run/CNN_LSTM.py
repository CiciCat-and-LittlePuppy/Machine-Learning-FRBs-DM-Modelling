import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time

start_time = time.time()

if 'SLURM_CPUS_PER_TASK' in os.environ:
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    print("number of cpus:", int(os.environ['SLURM_CPUS_PER_TASK']))
else:
    print("SLURM_CPUS_PER_TASK environment variable is not set. Using default number of threads.")

# -------------------------
# Model!!!!!!
# -------------------------
class FRB_CNN_LSTM_3Conv(nn.Module):
    def __init__(self, lstm_hidden_size=128, num_lstm_layers=2):
        super(FRB_CNN_LSTM_3Conv, self).__init__()

        # Conv1
        self.conv1 = nn.Conv1d(
            in_channels=512, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv2
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv3
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)

        # Conv4
        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers,
            batch_first=True, bidirectional=True
        )

        # Fully-connected
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.squeeze(1)  # (batch, 512, 1024)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Dataset
# -------------------------
class FRBDataset(Dataset):
    def __init__(self, npz_file_paths, base_dir=""):
        self.file_paths = npz_file_paths
        self.directory = base_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_paths[idx])
        data = np.load(file_path, allow_pickle=True)
        if "wfall" in data:
            image = data["wfall"]
        else:
            image = data["dm_time"]
        target = float(data["params"][0])
        image = self.transform(image)
        target = torch.tensor(target, dtype=torch.float32)
        return image, target, self.file_paths[idx]

def get_npz_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.npz')]

# -------------------------
# Parse Arguments
# -------------------------
parser = argparse.ArgumentParser(description="Train a 3-Conv 1D CNN + BiLSTM on FRB data (with full metrics).")
parser.add_argument('--train_data_dir', type=str, required=True, help='Directory for the training data')
parser.add_argument('--test_data_dir', type=str, required=True, help='Directory for the testing data')
parser.add_argument('--output_dir', type=str, required=True, help='Directory for outputs (checkpoints, logs, etc.)')
parser.add_argument('--plot_dir', type=str, required=True, help='Directory for plots')
parser.add_argument('--train_pred_dir', type=str, required=True, help='Directory for per-epoch training predictions')
args = parser.parse_args()

train_data_dir = os.path.expanduser(args.train_data_dir)
test_data_dir  = os.path.expanduser(args.test_data_dir)
output_dir     = os.path.expanduser(args.output_dir)
plot_dir       = os.path.expanduser(args.plot_dir)
train_pred_dir = os.path.expanduser(args.train_pred_dir)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(train_pred_dir, exist_ok=True)

print(f"Train data directory: {train_data_dir}")
print(f"Test data directory: {test_data_dir}")

# -------------------------
# Create Datasets / Loaders
# -------------------------
train_files = get_npz_files(train_data_dir)
test_files  = get_npz_files(test_data_dir)

train_dataset = FRBDataset(train_files, train_data_dir)
test_dataset  = FRBDataset(test_files, test_data_dir)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------
# Instantiate Model + Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FRB_CNN_LSTM_3Conv(lstm_hidden_size=128, num_lstm_layers=2).to(device)
criterion = nn.MSELoss()
mse_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)

num_epochs = 150
accumulation_steps = 1

# Tracking lists
train_mse_loss, test_mse_loss = [], []
train_mses, test_mses = [], []
train_rmses, test_rmses = [], []
train_maes, test_maes = [], []
train_good_rates, test_good_rates = [], []
train_merrps, test_merrps = [], []
train_good_errp_rates, test_good_errp_rates = [], []

error_table_path = os.path.join(output_dir, 'error_vs_epoch.txt')
with open(error_table_path, 'w') as f:
    f.write("Epoch\tTrain_MSELoss\tTest_MSELoss\tTrain_MSE\tTest_MSE\tTrain_RMSE\tTest_RMSE\t"
            "Train_MAE_2\tTest_MAE_2\tTrain_GoodRate\tTest_GoodRate\tTrain_MERRP\tTest_MERRP\t"
            "Train_GMERRP\tTest_GMERRP\n")

# -------------------------
# Training & Validation Loop
# -------------------------
for epoch in range(num_epochs):
    # ---- Per-epoch accumulators for predictions ----
    all_train_preds, all_train_tgts, all_train_filenames = [], [], []
    all_test_preds, all_test_tgts, all_test_filenames = [], [], []

    # ---- Train ----
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_abs = 0.0
    running_good = 0
    running_errp = 0.0
    running_good_errp = 0
    total_samples = 0

    optimizer.zero_grad()
    train_pred_file_path = os.path.join(train_pred_dir, f"{epoch}_train_preds.txt")
    with open(train_pred_file_path, 'w') as tpf:
        tpf.write("filename\tTrue_DM\tPred_DM\n")

    for i, (inputs, targets, filenames) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze()

        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()

        mse_val = mse_criterion(outputs, targets) / accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        batch_size = inputs.size(0)
        running_loss += loss.item() * accumulation_steps * batch_size
        running_mse   += mse_val.item() * accumulation_steps * batch_size

        preds = outputs.detach().cpu().numpy()
        labs = targets.detach().cpu().numpy()
        abs_diff = np.abs(preds - labs)
        abs_errp = abs_diff / labs
        running_abs += np.sum(abs_diff)
        running_good += np.sum(abs_diff < 1.427)
        running_errp += np.sum(abs_errp)
        running_good_errp += np.sum(abs_errp < 0.01)
        total_samples += batch_size

        all_train_preds.append(preds)
        all_train_tgts.append(labs)
        all_train_filenames.extend(filenames)

    all_train_preds = np.concatenate(all_train_preds)
    all_train_tgts = np.concatenate(all_train_tgts)
    with open(train_pred_file_path, 'a') as tpf:
        for name, tval, pval in zip(all_train_filenames, all_train_tgts, all_train_preds):
            tpf.write(f"{name}\t{tval:.8f}\t{pval:.8f}\n")

    # Train metrics
    epoch_train_error = running_loss / len(train_loader.dataset)
    epoch_train_mse = running_mse / len(train_loader.dataset)
    epoch_train_rmse = np.sqrt(epoch_train_mse)
    epoch_train_mae = running_abs / total_samples
    train_good_rate = running_good / total_samples
    train_merrp = running_errp / total_samples
    train_good_errp_rate = running_good_errp / total_samples

    train_mse_loss.append(epoch_train_error)
    train_mses.append(epoch_train_mse)
    train_rmses.append(epoch_train_rmse)
    train_maes.append(epoch_train_mae)
    train_good_rates.append(train_good_rate)
    train_merrps.append(train_merrp)
    train_good_errp_rates.append(train_good_errp_rate)

    # ---- Test ----
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_abs = 0.0
    running_good = 0
    running_errp = 0.0
    running_good_errp = 0
    total_samples = 0

    test_pred_file_path = os.path.join(train_pred_dir, f"{epoch}_test_preds.txt")
    with open(test_pred_file_path, 'w') as vpf:
        vpf.write("filename\tTrue_DM\tPred_DM\n")

    with torch.no_grad():
        for inputs, targets, filenames in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)
            mse_val = mse_criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_mse   += mse_val.item() * batch_size

            preds = outputs.detach().cpu().numpy()
            labs = targets.detach().cpu().numpy()
            abs_diff = np.abs(preds - labs)
            abs_errp = abs_diff / labs
            running_abs += np.sum(abs_diff)
            running_good += np.sum(abs_diff < 1.427)
            running_errp += np.sum(abs_errp)
            running_good_errp += np.sum(abs_errp < 0.01)
            total_samples += batch_size

            all_test_preds.append(preds)
            all_test_tgts.append(labs)
            all_test_filenames.extend(filenames)

    all_test_preds = np.concatenate(all_test_preds)
    all_test_tgts = np.concatenate(all_test_tgts)
    with open(test_pred_file_path, 'a') as tpf:
        for name, tval, pval in zip(all_test_filenames, all_test_tgts, all_test_preds):
            tpf.write(f"{name}\t{tval:.8f}\t{pval:.8f}\n")

    # Test metrics
    epoch_test_error = running_loss / len(test_loader.dataset)
    epoch_test_mse = running_mse / len(test_loader.dataset)
    epoch_test_rmse = np.sqrt(epoch_test_mse)
    epoch_test_mae = running_abs / total_samples
    test_good_rate = running_good / total_samples
    test_merrp = running_errp / total_samples
    test_good_errp_rate = running_good_errp / total_samples

    test_mse_loss.append(epoch_test_error)
    test_mses.append(epoch_test_mse)
    test_rmses.append(epoch_test_rmse)
    test_maes.append(epoch_test_mae)
    test_good_rates.append(test_good_rate)
    test_merrps.append(test_merrp)
    test_good_errp_rates.append(test_good_errp_rate)

    # ---- Print ----
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train MSE: {epoch_train_error:.4f}, Test MSE: {epoch_test_error:.4f}, "
          f"Train RMSE: {epoch_train_rmse:.4f}, Test RMSE: {epoch_test_rmse:.4f}, "
          f"Train MAE_2: {epoch_train_mae:.4f}, Test MAE_2: {epoch_test_mae:.4f}, "
          f"Train GDR: {train_good_rate:.4f}, Test GDR: {test_good_rate:.4f}, "
          f"Train MERRP: {train_merrp:.4f}, Test MERRP: {test_merrp:.4f}, "
          f"Train GMERRP: {train_good_errp_rate:.4f}, Test GMERRP: {test_good_errp_rate:.4f}")

    # Save checkpoint (last 100 epochs)
    if epoch >= (num_epochs - 100):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

    # Append to file
    with open(error_table_path, 'a') as f:
        f.write(
            f"{epoch+1}\t"
            f"{epoch_train_error:.4f}\t{epoch_test_error:.4f}\t"
            f"{epoch_train_mse:.4f}\t{epoch_test_mse:.4f}\t"
            f"{epoch_train_rmse:.4f}\t{epoch_test_rmse:.4f}\t"
            f"{epoch_train_mae:.4f}\t{epoch_test_mae:.4f}\t"
            f"{train_good_rate:.4f}\t{test_good_rate:.4f}\t"
            f"{train_merrp:.4f}\t{test_merrp:.4f}\t"
            f"{train_good_errp_rate:.4f}\t{test_good_errp_rate:.4f}\n"
        )

# -------------------------
# Plot
# -------------------------
plt.plot(range(1, num_epochs+1), np.log(train_mse_loss), label='Train MSE')
plt.plot(range(1, num_epochs+1), np.log(test_mse_loss), label='Test MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Testing MSE per Epoch - Version A')
plt.legend()
plot_filename = os.path.join(plot_dir, 'training_curve.png')
plt.savefig(plot_filename)
plt.close()

# Save final metrics to CSV
results_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train MSELoss': train_mse_loss,
    'Test MSELoss': test_mse_loss,
    'Train MSE': train_mses,
    'Test MSE': test_mses,
    'Train RMSE': train_rmses,
    'Test RMSE': test_rmses,
    'Train MAE_2': train_maes,
    'Test MAE_2': test_maes,
    'Train GoodRate': train_good_rates,
    'Test GoodRate': test_good_rates,
    'Train MERRP': train_merrps,
    'Test MERRP': test_merrps,
    'Train GMERRP': train_good_errp_rates,
    'Test GMERRP': test_good_errp_rates,
})
final_csv_path = os.path.join(output_dir, 'error_vs_epoch_final.csv')
results_df.to_csv(final_csv_path, index=False)
print(f"Final error table saved to {final_csv_path}")

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
