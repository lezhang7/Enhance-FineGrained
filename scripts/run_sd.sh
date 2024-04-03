#!/bin/bash
#SBATCH --job-name=sdgen_1Apr_rel
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/sdgen/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/sdgen/job_error-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo master port is $MASTER_POR
export WORLD_SIZE=$SLURM_NTASKS_PER_NODE
echo world size is $WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo master addr is $MASTER_ADDR
export OMP_NUM_THREADS=12

echo gpu_num is $SLURM_GPUS_ON_NODE




# train_data='../data/generated_data/coco_train/'
train_data='/network/projects/mair_project_vim/annotations/training_data/train.json'

lr=5e-06
bs=256
cd ./src

output_file=./Outputs/$SLURM_JOB_NAME

echo output file is $output_file
torchrun --master_port $MASTER_PORT  --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
    --train-data $train_data \
    --seed 42 \
    --dataset-type sdgen \
    --sole-hardnegative \
    --save-frequency 1 \
    --report-to wandb \
    --warmup 50 \
    --batch-size $bs \
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
    --wandb-project-name sd_gen_clip_modelling \
    --name $SLURM_JOB_NAME_$(date +'%d-%b-%Y-%H-%M-%S') \

