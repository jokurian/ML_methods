#!/bin/bash
#SBATCH --mem=60g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdkt-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=ef0
#SBATCH --time=12:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
#SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=jkurian@caltech.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options



# --- Define Variables ---
ntrain=20
filename="uccsd_t_result.xyz"

# --- Environment Setup ---
conda init
conda activate /u/jkurian/anaconda3/envs/uma

# --- Data Preparation ---
rm -r train val output
mkdir train val
python make_data.py --ntrain "$ntrain" --input_xyz "$filename"

#Change this to path of your fairchem. Might have to change the path inside the python script to path to fairchem too
python /projects/bdkt/jkurian/softwares/fairchem/src/fairchem/core/scripts/create_uma_finetune_dataset.py --train-dir ./train/ --val-dir ./val --output-dir ./output --uma-task omol --regression-task ef

# --- Finetune ---  #Change run_dir to some folder. Finetuned checkpoint file will be saved to a folder here
fairchem -c ./output/uma_sm_finetune_template.yaml epochs=2000 job.run_dir=/scratch/bdkt/jkurian/uma/ base_model_name=uma-s-1p1 batch_size=10 > finetune_output.out 2>&1

# --- Post-processing ---
mkdir result
cp finetune_output.out result
cd result

cp ../make_data.py .
cp ../"$filename" .

# Re-run data script without random error
python make_data.py --ntrain "$ntrain" --input_xyz "$filename"

# This post processing is not working as expected. Have to copy file from job.run_dir/checkpoint/final manually and then run_test.py. It will find the MAE of energy and force for the training and val data

# Get final model
#cp /projects/bdkt/jkurian/project/fairchem/withError/get_file.py .
#python get_file.py
#
## Run test
#cp /projects/bdkt/jkurian/project/fairchem/withError/run_test.py ./
#python run_test.py > run_output.out





