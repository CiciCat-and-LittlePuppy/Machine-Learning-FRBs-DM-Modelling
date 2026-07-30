#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        # Requesting 1 CPU per task
#SBATCH --mem=128G                 # Requesting 128GB memory
#SBATCH --time=00-24:00          # Shorter job time for test, e.g., 5 minutes
#SBATCH --output=test_job_%j.out  # Standard output and error log for test

module load python/3.10
source frb/bin/activate
module load scipy-stack
pip install torch torchvision sklearn --no-index
for i in {1..9}
do
    python project_path/generating_full_cover_data/run_frb_simulationv1.34.py --output_dir="project_path/super_good_data/$i"
done