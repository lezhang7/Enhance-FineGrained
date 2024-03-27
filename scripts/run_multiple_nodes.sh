#!/bin/bash
#SBATCH --job-name=4node
#SBATCH --partition=long                          # Ask for unkillable job
#SBATCH --cpus-per-task=6                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --time=8:00:00                                     
#SBATCH --output=~/slurm_logs/Enhance-FineGrained/case11_output-%j.txt
#SBATCH --error=~/slurm_logs/Enhance-FineGrained/case11_error-%j.txt 

# load modules for slurm
module load miniconda/3
conda init
conda activate openflamingo

# multinode setup
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo master port is $MASTER_POR
export WORLD_SIZE=$SLURM_NTASKS_PER_NODE
echo world size is $WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo master addr is $MASTER_ADDR
export OMP_NUM_THREADS=12
echo gpu_num is $SLURM_GPUS_ON_NODE

# set up data
IMG_DATA_DIR=~/dataset/coco/2014/train2014.zip # set path to coco 2014 folder
mkdir $SLURM_TMPDIR/data
unzip -q $IMG_DATA_DIR -d $SLURM_TMPDIR/data
echo "unzip done"
export IMG_DATA_DIR=$SLURM_TMPDIR/data

train_data='../data/generated_data/coco_train/'
upper_bound=10
threshold_type=mean # fixed or mean
fixed_threshold_value=10
lr=5e-06
bs=378
cd ./src


for cmr_weight in 0.2
do
    for imc_weight in 0.3
    do
        if [ "$threshold_type" == "fixed" ]; then
            output_name=coco_hn_imc$imc_weight-cmr$cmr_weight-fixed$fixed_threshold_value-$lr
        else
            output_name=coco_hn_imc$imc_weight-cmr$cmr_weight-mean-ub$upper_bound-$lr
        fi
        output_file=./Outputs/$output_name

        if [[ -d "$output_file" ]];then
            echo "$output_name already exists"
        else
            echo "running $output_name"
            torchrun --master_port $MASTER_PORT  --nproc_per_node=$SLURM_GPUS_ON_NODE \
            main.py \
            --wandb-project-name open_clip \
            --train-data $train_data \
            --seed 42 \
            --dataset-type npy \
            --save-frequency 1 \
            --report-to wandb \
            --warmup 50 \
            --batch-size $bs \
            --lr $lr \
            --wd 0.1 \
            --precision amp \
            --epochs 10 \
            --workers 0 \
            --pretrained openai \
            --model ViT-B-32 \
            --logs Outputs \
            --beta1 0.9 \
            --beta2 0.98 \
            --eps 1e-06 \
            --log-every-n-steps 10 \
            --imc-loss \
            --cmr-loss \
            --hardnegative \
            --threshold-type $threshold_type \
            --fixed-threshold-value $fixed_threshold_value \
            --cmr-loss-weight $cmr_weight \
            --imc-loss-weight $imc_weight \
            --upper-bound $upper_bound \
            --name $output_name

            # Check if the training command was successful
            if [ $? -ne 0 ]; then
                echo "Training failed. Cleaning up..."
                # Delete the output folder
                rm -rf $output_file
            fi
        fi
    done
done

