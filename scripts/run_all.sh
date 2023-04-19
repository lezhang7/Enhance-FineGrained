#!/bin/bash
#SBATCH --job-name=all_cc
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=16                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --ntasks-per-node=2                                  # Ask for 1 GPU
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

upper_bound=5
rankweight=0.2
disweight=0.2
lr=5e-06
bs=256
cd /home/mila/l/le.zhang/scratch/open_clip/src
for rankweight in 0.2 0.3 0.4 0.5 0.7 0.9 1.0
do
    for disweight in 0.2 0.3 0.4 0.5 0.7 0.9 1.0
    do
        output_name=rank_coco-dis_text_mean-hn--$lr-weightd$disweight-weightr$rankweight-ub$upper_bound-w_special
        output_file=/home/mila/l/le.zhang/scratch/open_clip/src/Outputs/$output_name
        if [[ -d "$output_file" ]];then
            echo "$output_name already exists"
            
        else
            echo "running $output_name"
            echo "$output_name" >> /home/mila/l/le.zhang/scratch/vision-language-models-are-bows/experiments/run_all_hypertuning_names_coco
            torchrun --master_port $MASTER_PORT  --nproc_per_node=2 \
            -m training.main \
            --wandb-project-name open_clip \
            --train-data '/home/mila/l/le.zhang/scratch/open_clip/processed_dataset' \
            --val-data winoground \
            --seed 42 \
            --dataset-type npy \
            --save-frequency 1 \
            --report-to wandb \
            --warmup 50 \
            --batch-size=$bs \
            --lr $lr \
            --wd 0.1 \
            --epochs 10 \
            --workers 0 \
            --pretrained openai \
            --model ViT-B-32 \
            --logs Outputs \
            --beta1 0.9 \
            --beta2 0.98 \
            --eps 1e-06 \
            --log-every-n-steps 10 \
            --discriminative-loss text \
            --rank-loss \
            --threshold mean \
            --rank-loss-weight $rankweight \
            --dis-loss-weight $disweight \
            --hardnegative \
            --upper-bound $upper_bound \
            --name $output_name

        fi
    done
done