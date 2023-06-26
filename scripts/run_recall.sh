#!/bin/bash
#SBATCH --job-name=cal_recall
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G                                        # Ask for 10 GB of RAM
#SBATCH --time=00:10:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/job_error-%j.txt 

module load miniconda/3
conda init
conda activate aro
cd /home/mila/l/le.zhang/scratch/open_clip/src
python test_compute_recall.py