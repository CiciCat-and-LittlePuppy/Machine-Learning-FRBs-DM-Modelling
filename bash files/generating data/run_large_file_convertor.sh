#!/bin/bash
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1        # Requesting 1 CPUs per task
#SBATCH --mem=256G               # Requesting 256GB memory
#SBATCH --time=00-01:00          # Job time limit: 6 hours
#SBATCH --output=test_job_%j.out # Standard output and error log for test

module load python/3.10
source frb/bin/activate
module load scipy-stack
pip install torch torchvision sklearn --no-index

# Start time
START_TIME=$(date +%s)

# Create a temporary directory for the converted data
mkdir -p $SLURM_TMPDIR/converted_data_good_full_cover

# Run the Python script with specified directories
python /home/beizi/projects/def-rajabf1/beizi/single_pulse_ml/generating_full_cover_data/large_file_convertor.py \
    --output_dir=$SLURM_TMPDIR/converted_data_good_full_cover \
    --data_dir=/home/beizi/projects/def-rajabf1/beizi/single_pulse_ml/Inference_data_2 \
    --exclude_dir=/home/beizi/projects/def-rajabf1/beizi/single_pulse_ml/Inference_data_2/excluding

# Change to the temporary directory
cd $SLURM_TMPDIR/converted_data_good_full_cover
# Create a tar archive of the converted data
tar -cf Inference_data.tar .

# Copy the tar archive to the final destination
cp $SLURM_TMPDIR/converted_data_good_full_cover/Inference_data.tar /home/beizi/projects/def-rajabf1/beizi/single_pulse_ml/Inference_data_2

# End time
END_TIME=$(date +%s)

# Calculate duration
DURATION=$((END_TIME - START_TIME))
# Output running time
echo "Job completed in $DURATION seconds."