#!/bin/bash

module load miniconda/3
conda init
conda activate aro


cd /home/mila/l/le.zhang/scratch/VALSE
bs=64
model_name="xvlm-coco"
output_dir="./output"


checkpoint=Enhance-FineGrained/src/Outputs/$resume_name/checkpoints/epoch_$resume_n.pt           
python3 xvlm_valse_eval.py --output-dir $output_dir --batch-size $bs --model-name $model_name  --resume $resume





