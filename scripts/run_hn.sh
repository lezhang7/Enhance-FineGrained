#!/bin/bash
#SBATCH --job-name=coco_hn
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=16                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --reservation=DGXA100
#SBATCH --ntasks-per-node=4                                # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=120:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/open_clip/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/open_clip/slurm_logs/job_error-%j.txt 

module load miniconda/3
conda init
conda activate flava
echo start !

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$((${SLURM_JOB_NUM_NODES:=1} * $SLURM_GPUS_ON_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export OMP_NUM_THREADS=12

disweight=0.2
lr=5e-06
#5e-06 1e-06 
cd /home/mila/l/le.zhang/scratch/open_clip/src

torchrun --master_port $MASTER_PORT  --nproc_per_node=4 \
    -m training.main \
    --wandb-project-name open_clip \
    --train-data '/home/mila/l/le.zhang/scratch/winonground/data/processed_dataset/coco/' \
    --val-data winoground \
    --seed 42 \
    --dataset-type npy \
    --save-frequency 1 \
    --report-to wandb \
    --warmup 50 \
    --batch-size=128 \
    --lr $lr \
    --wd 0.1 \
    --epochs 5 \
    --workers 0 \
    --pretrained openai \
    --model ViT-B-32 \
    --logs Outputs \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-06 \
    --log-every-n-steps 10 \
    --hardnegative \
    --name clip_coco-hn-$lr-weight$disweight
