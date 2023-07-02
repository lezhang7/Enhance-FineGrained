#!/bin/bash

module load miniconda/3
conda init
conda activate aro


cd /home/mila/l/le.zhang/scratch/VALSE
bs=64
model_name="blip-coco-base"
output_dir="./output"

python3 xvlm_valse_eval.py --output-dir $output_dir --batch-size $bs --model-name $model_name 




