#!/bin/bash
#SBATCH --job-name=eval_itrhn
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G                                        # Ask for 10 GB of RAM
#SBATCH --time=50:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/job_error-%j.txt 


module load miniconda/3
conda init
conda activate flava
echo start
bs=5160
cd /home/mila/l/le.zhang/scratch/open_clip/src
while read checkpoint; do
checkpoint="/home/mila/l/le.zhang/scratch/open_clip/src/Outputs/$checkpoint"
echo $checkpoint
    python3 eval_recall.py \
    --pretrained $checkpoint \
    --batch_size $bs \
    --size 20000 \
    --output-dir './itr_results/' \
    --dataset '/home/mila/l/le.zhang/scratch/winonground/data/processed_dataset/coco_val' 
done </home/mila/l/le.zhang/scratch/open_clip/src/itr_results/checkpoints.txt