#!/bin/bash
#SBATCH --job-name=aro_test
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                   
#SBATCH --output=/home/mila/l/le.zhang/scratch/slurm_logs/job_output-%j.txt
#SBATCH --error=/home/mila/l/le.zhang/scratch/slurm_logs/job_error-%j.txt 

module load miniconda/3
conda init
conda activate aro

bs=1024
# model=openai-clip:ViT-B/32  
# model=xvlm-coco
# model=NegCLIP
model=ours
checkpoint=vision-language-models-are-bows/~/.cache/itchn_atr0.2_mean_tec0.3/checkpoint_epoch_0.pt


cd Enhance-FineGrained/vision-language-models-are-bows


for dataset in VG_Relation VG_Attribution COCO_Retrieval
do
    python3 main_aro.py --dataset=$dataset --model-name=$model --device=cuda --batch-size $bs --resume=$checkpoint
done