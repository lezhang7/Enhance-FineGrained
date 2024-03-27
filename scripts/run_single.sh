#!/bin/bash
#SBATCH --job-name=test_single
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=4                                # Ask for 1 GPU
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=120:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/Enhance-FineGrained/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/Enhance-FineGrained/job_error-%j.txt 



module load miniconda/3
conda init
conda activate aro



train_data='../data/generated_data/coco_train/'
upper_bound=10
threshold_type=mean # fixed or mean
fixed_threshold_value=10
lr=5e-06
bs=256
cd ./src

for atr_weight in 0.2
do
    for tec_weight in 0.3
    do
        if [ "$threshold_type" == "fixed" ]; then
            output_name=coco_hn_tec$tec_weight-atr$atr_weight-fixed$fixed_threshold_value-$lr
        else
            output_name=coco_hn_tec$tec_weight-atr$atr_weight-mean-ub$upper_bound-$lr-test_single
        fi
        output_file=./Outputs/$output_name

        if [[ -d "$output_file" ]];then
            echo "$output_name already exists"
        else
            echo "running $output_name"
            python main.py \
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
            --epochs 10 \
            --workers 0 \
            --pretrained openai \
            --model ViT-B-32 \
            --logs Outputs \
            --beta1 0.9 \
            --precision amp \
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
            fi
        fi
    done
done

