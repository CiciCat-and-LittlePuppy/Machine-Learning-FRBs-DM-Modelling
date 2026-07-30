#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=2        # Requesting 2 CPUs per task
#SBATCH --mem=128G               # Requesting 128GB memory
#SBATCH --time=00-124:00          # Requesting 10 hours (day-hour:minute format)
#SBATCH --output=test_job_%j.out # Standard output and error log for test

module load python/3.10
source environment_path/bin/activate
module load scipy-stack

# Install torch, torchvision, and sklearn from the default sources
pip install torch torchvision sklearn

# Install config separately, for example from a local file
pip install --no-index --find-links="project_path/package" config

# Setup temporary data directory
mkdir -p "$SLURM_TMPDIR/data_CNN"
mkdir -p "$SLURM_TMPDIR/test_CNN"

cp "path to data storage/train_dataset.tar" "$SLURM_TMPDIR/data_CNN"

cp "path to data storage/test_dataset.tar" "$SLURM_TMPDIR/test_CNN"

cd "$SLURM_TMPDIR/data_CNN"

tar -xf train_dataset.tar

cd "$SLURM_TMPDIR/test_CNN"

tar -xf test_dataset.tar

# Run the Python script
python project_path/CNN_faster.py \
    --train_data_dir="$SLURM_TMPDIR/data_CNN" \
    --test_data_dir="$SLURM_TMPDIR/test_CNN" \
    --plot_dir=plot_path \
    --output_dir=checkpoint_path \
    --train_pred_dir=model_prediction_path \
    --table_dir=result_conclusion_path