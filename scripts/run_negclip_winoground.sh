#!/bin/bash
#SBATCH --job-name=eval_negclip_winoground
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=16                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 
#SBATCH --reservation=DGXA100
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=120:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/open_clip/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/open_clip/slurm_logs/job_error-%j.txt 

module load miniconda/3
conda init
conda activate flava
echo start !

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export OMP_NUM_THREADS=12

rankweight=0.2
lr=5e-06
resume=/home/mila/l/le.zhang/scratch/open_clip/src/Outputs/negclip/checkpoints/negCLIP.pt
cd /home/mila/l/le.zhang/scratch/open_clip/src
torchrun --master_port $MASTER_PORT  --nproc_per_node=1 \
    -m training.main \
    --val-data winoground \
    --seed 42 \
    --dataset-type npy \
    --report-to "tensorboard" \
    --batch-size=128 \
    --workers 0 \
    --pretrained=$resume \
    --model ViT-B-32 \
    --logs Outputs \
    --name negclip_winoground_0