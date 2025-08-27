#!/bin/bash
#SBATCH --mem=60g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdkt-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=ef1
#SBATCH --time=12:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
#SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=jkurian@caltech.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options


conda init
conda activate /u/jkurian/anaconda3/envs/uma

# --- Define Variables ---
ntrain=40
error_value=0.04336  #in eV
filename="uccsd_t_result.xyz"

# --- Environment Setup ---
conda init
conda activate /u/jkurian/anaconda3/envs/uma

# --- Data Preparation ---
mkdir train val
python make_data.py --ntrain "$ntrain" --add_random_error --error_value "$error_value" --input_xyz "$filename"

#Change --regression-task to e to just train with energy
python /projects/bdkt/jkurian/softwares/fairchem/src/fairchem/core/scripts/create_uma_finetune_dataset.py --train-dir ./train/ --val-dir ./val --output-dir ./output --uma-task omol --regression-task ef

# --- Finetune ---
fairchem -c ./output/uma_sm_finetune_template.yaml epochs=2000 job.run_dir=/scratch/bdkt/jkurian/uma/ base_model_name=uma-s-1p1 batch_size=10 > finetune_output.out 2>&1

# --- Post-processing ---
mkdir result
cp finetune_output.out result
cd result

cp ../make_data.py .
cp ../"$filename" .

# Re-run data script without random error for testing
python make_data.py --ntrain "$ntrain"  --input_xyz "$filename"

#Do manually: copy checkpoint file from run_dir/checkpoints/final
#python run_test.py

## Get final model
#cp ../get_file.py .
#python get_file.py
#
## Run test
#cp ../run_test.py ./
#python run_test.py > run_output.out






