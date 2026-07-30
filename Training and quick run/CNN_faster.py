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

class FRBnet(nn.Module):
    def __init__(self):
        super(FRBnet, self).__init__()

        # Convolutional layers + BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5   = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6   = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.bn7   = nn.BatchNorm2d(2048)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        flattened_size = 7 * 7 * 2048

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 16384)
        self.fc2 = nn.Linear(16384, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1)

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
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def get_npz_files(directory):
    return [file for file in os.listdir(directory) if file.endswith('.npz')]

parser = argparse.ArgumentParser(description='FRB Net')
parser.add_argument('--train_data_dir', type=str, required=True, help='Directory for the training data')
parser.add_argument('--test_data_dir', type=str, required=True, help='Directory for the testing data')
parser.add_argument('--output_dir', type=str, required=True, help='Directory for the outputs')
parser.add_argument('--plot_dir', type=str, required=True, help='Directory for the plots')
parser.add_argument('--train_pred_dir', type=str, required=True, help='Directory for the training prediction')
parser.add_argument('--table_dir', type=str, required=True, help='Directory for the training prediction')
args = parser.parse_args()

train_data_dir = os.path.expanduser(args.train_data_dir)
test_data_dir  = os.path.expanduser(args.test_data_dir)
output_dir = os.path.expanduser(args.output_dir)
plot_dir = os.path.expanduser(args.plot_dir)
train_pred_dir = os.path.expanduser(args.train_pred_dir)
table_dir = os.path.expanduser(args.table_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(train_pred_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

train_files = get_npz_files(train_data_dir)
test_files  = get_npz_files(test_data_dir)

class FRBDataset(Dataset):
    def __init__(self, npz_file_paths, base_dir=""):
        self.file_paths = npz_file_paths
        self.directory = base_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        fname = self.file_paths[idx]
        file_path = os.path.join(self.directory, self.file_paths[idx])
        data = np.load(file_path, allow_pickle=True)
        image = data["wfall"] if "wfall" in data.keys() else data["dm_time"]
        target = np.array(float(data['params'][0]))
        image = self.transform(image)  # shape: (1,224,224)
        target = torch.from_numpy(target).float()
        return image, target, fname

train_dataset = FRBDataset(train_files, train_data_dir)
test_dataset  = FRBDataset(test_files, test_data_dir)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = FRBnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print(f"There are {torch.cuda.device_count()} GPUs! Yeah!!!!!!")
    model = nn.DataParallel(model)

model = model.to(device)

criterion = nn.L1Loss()
mse_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
num_epochs = 60

train_mae_loss, test_mae_loss = [], []
train_mses, test_mses = [], []
train_rmses, test_rmses = [], []
train_maes, test_maes = [], []
train_good_rates, test_good_rates = [], []
train_merrps, test_merrps = [], []
train_good_errp_rates, test_good_errp_rates = [], []

if torch.cuda.is_available():
    print("model")
    print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9} GB")

error_table_path = os.path.join(table_dir, 'error_vs_epoch.txt')
with open(error_table_path, 'w') as f:
    f.write("Epoch\tTrain_L1Loss\tTest_L1Loss\tTrain_MSE\tTest_MSE\tTrain_RMSE\tTest_RMSE\t"
            "Train_MAE_2\tTest_MAE_2\tTrain_GoodRate\tTest_GoodRate\tTrain_MERRP\tTest_MERRP\tTrain_GERRP_Rate\tTest_GERRP_rate\n")

for epoch in range(num_epochs):
    all_train_preds, all_train_tgts, all_train_filenames = [], [], []
    all_test_preds, all_test_tgts, all_test_filenames = [], [], []

    optimizer.zero_grad()
    train_pred_file_path = os.path.join(train_pred_dir, f"{epoch}_train_preds.txt")
    with open(train_pred_file_path, 'w') as tpf:
        tpf.write("filename\tTrue_DM\tPred_DM\n")
    test_pred_file_path = os.path.join(train_pred_dir, f"{epoch}_test_preds.txt")
    with open(test_pred_file_path, 'w') as vpf:
        vpf.write("filename\tTrue_DM\tPred_DM\n")

    # Training
    model.train()
    running_loss, running_mse, running_abs = 0.0, 0.0, 0.0
    running_errp, running_good, running_good_errp, total_samples = 0.0, 0, 0, 0

    for i, (inputs, targets, filename) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        mse_val = mse_criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_mse += mse_val.item() * batch_size

        preds = outputs.detach().cpu().numpy()
        labs = targets.detach().cpu().numpy()
        abs_diff = np.abs(preds - labs)
        abs_errp = abs_diff / labs
        running_abs += np.sum(abs_diff)
        running_good += np.sum(abs_diff < 1.427)
        running_good_errp += np.sum(abs_errp < 0.01)
        running_errp += np.sum(abs_errp)
        total_samples += batch_size

        # Save batch predictions
        all_train_preds.append(preds)
        all_train_tgts.append(labs)
        all_train_filenames.extend(filename)

    all_train_preds = np.concatenate(all_train_preds)
    all_train_tgts = np.concatenate(all_train_tgts)
    # Save all train predictions
    with open(train_pred_file_path, 'a') as tpf:
        for name, tval, pval in zip(all_train_filenames, all_train_tgts, all_train_preds):
            tpf.write(f"{name}\t{tval:.8f}\t{pval:.8f}\n")

    # Metrics
    epoch_train_l1 = running_loss / len(train_loader.dataset)
    train_mae_loss.append(epoch_train_l1)
    epoch_train_mse = running_mse / len(train_loader.dataset)
    epoch_train_rmse = np.sqrt(epoch_train_mse)
    train_mses.append(epoch_train_mse)
    train_rmses.append(epoch_train_rmse)
    epoch_train_mae = running_abs / total_samples
    train_maes.append(epoch_train_mae)
    train_good_rate = running_good / total_samples
    train_good_rates.append(train_good_rate)
    train_merrp = running_errp / total_samples
    train_merrps.append(train_merrp)
    train_good_errp_rate = running_good_errp / total_samples
    train_good_errp_rates.append(train_good_errp_rate)

    # Validation
    model.eval()
    running_loss, running_mse, running_abs = 0.0, 0.0, 0.0
    running_errp, running_good, running_good_errp, total_samples = 0.0, 0, 0, 0
    with torch.no_grad():
        for inputs, targets, filename in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            mse_val = mse_criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_mse += mse_val.item() * batch_size

            preds = outputs.detach().cpu().numpy()
            labs = targets.detach().cpu().numpy()
            abs_diff = np.abs(preds - labs)
            abs_errp = abs_diff / labs
            running_abs += np.sum(abs_diff)
            running_good += np.sum(abs_diff < 1.427)
            running_good_errp += np.sum(abs_errp < 0.01)
            running_errp += np.sum(abs_errp)
            total_samples += batch_size

            # Save batch predictions
            all_test_preds.append(preds)
            all_test_tgts.append(labs)
            all_test_filenames.extend(filename)

    all_test_preds = np.concatenate(all_test_preds)
    all_test_tgts = np.concatenate(all_test_tgts)
    # Save all test predictions
    with open(test_pred_file_path, 'a') as tpf:
        for name, tval, pval in zip(all_test_filenames, all_test_tgts, all_test_preds):
            tpf.write(f"{name}\t{tval:.8f}\t{pval:.8f}\n")

    # Metrics
    epoch_test_l1 = running_loss / len(test_loader.dataset)
    test_mae_loss.append(epoch_test_l1)
    epoch_test_mse = running_mse / len(test_loader.dataset)
    epoch_test_rmse = np.sqrt(epoch_test_mse)
    test_mses.append(epoch_test_mse)
    test_rmses.append(epoch_test_rmse)
    epoch_test_mae = running_abs / total_samples
    test_maes.append(epoch_test_mae)
    test_good_rate = running_good / total_samples
    test_good_rates.append(test_good_rate)
    test_merrp = running_errp / total_samples
    test_merrps.append(test_merrp)
    test_good_errp_rate = running_good_errp / total_samples
    test_good_errp_rates.append(test_good_errp_rate)

    print(f"Epoch {epoch+1}, "
          f"Train L1: {epoch_train_l1:.4f}, Test L1: {epoch_test_l1:.4f}, "
          f"Train MSE: {epoch_train_mse:.4f}, Test MSE: {epoch_test_mse:.4f}, "
          f"Train RMSE: {epoch_train_rmse:.4f}, Test RMSE: {epoch_test_rmse:.4f}, "
          f"Train MAE_2: {epoch_train_mae:.4f}, Test MAE_2: {epoch_test_mae:.4f}, "
          f"Train GDR: {train_good_rate:.4f}, Test GDR: {test_good_rate:.4f},"
          f"Train MERRP: {train_merrp:.4f}, Test MERRP: {test_merrp:.4f},"
          f"Train GMERRPR: {train_good_errp_rate:.4f}, Test GMERRPR: {test_good_errp_rate:.4f}")

    # Save checkpoint for the last 60 epochs
    if epoch >= (num_epochs - 60):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

    with open(error_table_path, 'a') as f:
        f.write(
            f"{epoch+1}\t"
            f"{epoch_train_l1:.4f}\t{epoch_test_l1:.4f}\t"
            f"{epoch_train_mse:.4f}\t{epoch_test_mse:.4f}\t"
            f"{epoch_train_rmse:.4f}\t{epoch_test_rmse:.4f}\t"
            f"{epoch_train_mae:.4f}\t{epoch_test_mae:.4f}\t"
            f"{train_good_rate:.4f}\t{test_good_rate:.4f}\t"
            f"{train_merrp:.4f}\t{test_merrp:.4f}\t"
            f"{train_good_errp_rate:.4f}\t{test_good_errp_rate:.4f}\n"
        )

# Plot the L1Loss (MAE_1) over epochs
plt.plot(range(1, num_epochs+1), np.log(np.array(train_mae_loss)+1e-12), label='Train MAE (L1)')
plt.plot(range(1, num_epochs+1), np.log(np.array(test_mae_loss)+1e-12), label='Test MAE (L1)')
plt.xlabel('Epoch')
plt.ylabel('MAE (L1Loss)')
plt.title('Training and Testing MAE per Epoch (L1Loss)')
plt.legend()
plot_filename = os.path.join(plot_dir, 'training_curve.png')
plt.savefig(plot_filename)

results_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train L1Loss': train_mae_loss,
    'Test L1Loss': test_mae_loss,
    'Train MSE': train_mses,
    'Test MSE': test_mses,
    'Train RMSE': train_rmses,
    'Test RMSE': test_rmses,
    'Train MAE_2': train_maes,
    'Test MAE_2': test_maes,
    'Train GoodRate': train_good_rates,
    'Test GoodRate': test_good_rates,
    'Train Merrp': train_merrps,
    'Test Merrp': test_merrps,
    'Train Good Merrp Rate': train_good_errp_rates,
    'Test Good Merrp Rate': test_good_errp_rates,
})
final_csv_path = os.path.join(output_dir, 'error_vs_epoch_final.csv')
results_df.to_csv(final_csv_path, index=False)
print(f"Final error table saved to {final_csv_path}")

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")