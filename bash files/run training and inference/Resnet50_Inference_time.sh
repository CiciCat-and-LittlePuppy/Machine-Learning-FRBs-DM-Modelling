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

mkdir -p "$SLURM_TMPDIR/res-4_validation"  # -p added for consistency

cp data_path/Inference_data.tar \
   "$SLURM_TMPDIR/ResNet50_inference"

cd "$SLURM_TMPDIR/ResNet50_inference"
tar -xf Inference_data.tar

python project_path/Time_Inference_New/res50.py \
    --val_data_dir="$SLURM_TMPDIR/ResNet50_Inference" \
    --folds_base_dir=project_path/model_weights/res50_weights \
    --batch_size=16