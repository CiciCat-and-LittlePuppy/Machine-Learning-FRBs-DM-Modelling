#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        # Standardised to 1 across all three jobs
#SBATCH --mem=128G
#SBATCH --time=00-00:36
#SBATCH --output=test_job_%j.out

module load python/3.10
source frb/bin/activate
module load scipy-stack

pip install torch torchvision sklearn
pip install --no-index --find-links=project_path/package config

mkdir -p "$SLURM_TMPDIR/CNN_Inference"

cp data_path/Inference_data.tar \
   "$SLURM_TMPDIR/CNN_Inference"

cd "$SLURM_TMPDIR/CNN_Inference"
tar -xf Inference_data.tar

python project_path/Time_Inference_New/CNN.py \
    --checkpoint_dir=project_path/model_weights/CNN_weights \
    --validation_data_dir="$SLURM_TMPDIR/CNN_Inference" \
    --output_dir=output_path \
    --batch_size=16