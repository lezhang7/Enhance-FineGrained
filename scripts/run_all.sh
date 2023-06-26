#!/bin/bash
#SBATCH --job-name=all_coco_hyper
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2                                # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   

module load miniconda/3
conda init
conda activate flava

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo master port is $MASTER_POR
export WORLD_SIZE=$SLURM_NTASKS_PER_NODE
echo world size is $WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo master addr is $MASTER_ADDR
export OMP_NUM_THREADS=12

echo gpu_num is $SLURM_GPUS_ON_NODE


train_data='../data/generated_data/coco_train/'
upper_bound=10
atr_weight=0.2
tec_weight=0.2
threshold_type=mean # fixed or mean
fixed_threshold_value=5
lr=5e-06
bs=128
cd ../src/

for atr_weight in 0.2 0.4 0.6 0.8 1.0
do
    for tec_weight in 0.2 0.4 0.6 0.8 1.0
    do
        if [ "$threshold_type" == "fixed" ]; then
            output_name=coco_hn_tec$tec_weight-atr$atr_weight-fixed$fixed_threshold_value-$lr
        else
            output_name=coco_hn_tec$tec_weight-atr$atr_weight-mean-ub$upper_bound-$lr
        fi
        output_file=./Outputs/$output_name

        if [[ -d "$output_file" ]];then
            echo "$output_name already exists"
        else
            echo "running $output_name"
            echo "$output_name" >> ../experiments/run_all_hypertuning_names_coco
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
            --epochs 5 \
            --workers 0 \
            --pretrained openai \
            --model ViT-B-32 \
            --logs Outputs \
            --beta1 0.9 \
            --beta2 0.98 \
            --eps 1e-06 \
            --log-every-n-steps 10 \
            --tec-loss \
            --atr-loss \
            --hardnegative \
            --threshold-type $threshold_type \
            --fixed-threshold-value $fixed_threshold_value \
            --atr-loss-weight $atr_weight \
            --tec-loss-weight $tec_weight \
            --upper-bound $upper_bound \
            --name $output_name

            # Check if the training command was successful
            if [ $? -ne 0 ]; then
                echo "Training failed. Cleaning up..."
                # Delete the output folder
                rm -rf $output_file
                # Remove the output name from the file
                sed -i "/$output_name/d" ../experiments/run_all_hypertuning_names_coco
            fi
        fi
    done
done

