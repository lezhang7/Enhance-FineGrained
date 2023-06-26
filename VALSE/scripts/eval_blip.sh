#!/bin/bash
#SBATCH --job-name=aro_test
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=20:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/job_error-%j.txt 


module load miniconda/3
conda init
conda activate aro


cd /home/mila/l/le.zhang/scratch/VALSE
bs=64
model_name="blip-coco-base"
output_dir="./output"

python3 xvlm_valse_eval.py --output-dir $output_dir --batch-size $bs --model-name $model_name 




