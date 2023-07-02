#!/bin/bash

module load miniconda/3
conda init
conda activate aro

cd /home/mila/l/le.zhang/scratch/Enhance-FineGrained/VALSE

checkpoint=/home/mila/l/le.zhang/scratch/Enhance-FineGrained/clip/clip_all.pt           
python3 clip_valse_eval.py --pretrained $checkpoint --output-dir ./output    





