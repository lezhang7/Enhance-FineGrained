#!/bin/bash
#SBATCH --job-name=sdgen_1Apr_object
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=~/slurm_logs/sdgen/job_output-%j.txt
#SBATCH --error=~/slurm_logs/sdgen/job_error-%j.txt 
module load miniconda/3
conda init
conda activate openflamingo




# train_data='../data/generated_data/coco_train/'
train_data='/home/mila/l/le.zhang/scratch/datasets/minimal_change/Apr/train_data.json'
val_data='/home/mila/l/le.zhang/scratch/datasets/minimal_change/Apr/val_data.json'
lr=5e-06
bs=256
cd ./src

output_file=./Outputs/test
echo output file is $output_file

python main.py \
    --train-data $train_data \
    --val-data $val_data \
    --train-num-samples 1000 \
    --seed 42 \
    --dataset-type sdgen \
    --categories object \
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
    --name test_$(date +'%d-%b-%Y-%H-%M-%S') \

if [ $? -ne 0 ]; then
    echo "Training failed. Cleaning up..."
    # Delete the output folder
    rm -rf $output_file
    # Remove the output name from the file
fi

