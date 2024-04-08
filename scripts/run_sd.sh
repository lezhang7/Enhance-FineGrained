#!/bin/bash
#SBATCH --job-name=siglip_Apr1_all
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/sdgen/job_output-%j.txt
#SBATCH --error=./slurm_logs/sdgen/job_error-%j.txt 

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
train_data='/home/mila/l/le.zhang/scratch/datasets/minimal_change/Apr/train_data.json'
val_data='/home/mila/l/le.zhang/scratch/datasets/minimal_change/Apr/val_data.json'

lr=5e-06
bs=372
cd ./src

job_name=$SLURM_JOB_NAME
current_time=$(date +'%d-%b-%Y-%H-%M-%S')
output_name="${job_name}_${current_time}"
output_file="./Outputs/${output_name}"

# --sole-hardnegative # randomly sample an edited image-text pair for each sample
# --categories object # to include object category samples | selection from [object, attribute, relation, counting]
# --categories object attribute # to include object and attribute category samples
echo output path is $output_file

export NCCL_DEBUG=INFO

torchrun --master_port $MASTER_PORT  --nproc_per_node=$SLURM_GPUS_ON_NODE main.py \
    --train-data $train_data \
    --val-data $val_data \
    --seed 42 \
    --dataset-type sdgen \
    --save-frequency 1 \
    --report-to wandb \
    --warmup 50 \
    --batch-size $bs \
    --lr $lr \
    --wd 0.1 \
    --epochs 10 \
    --workers 0 \
    --model siglip \
    --pretrained google/siglip-base-patch16-224 \
    --logs Outputs \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-06 \
    --log-every-n-steps 10 \
    --wandb-project-name sd_gen_clip_modelling \
    --name $output_name \

if [ $? -ne 0 ]; then
    echo "Training failed. Cleaning up..."
    # Delete the output folder
    rm -rf $output_file
    # Remove the output name from the file
fi
