#!/bin/bash

module load miniconda/3
conda init
conda activate aro

bs=1024
# model=openai-clip:ViT-B/32  
# model=xvlm-coco
# model=NegCLIP
model=ours
checkpoint=/home/mila/l/le.zhang/scratch/Enhance-FineGrained/clip/clip_all.pt


cd ../

# VG-R AND VG-A
for dataset in VG_Relation VG_Attribution 
do
    python3 main_aro.py --dataset=$dataset --model-name=$model --device=cuda --batch-size=$bs --resume=$checkpoint --download  
done
# Standard COCO Retrieval
python3 main_retrieval.py --dataset=COCO_Retrieval --model-name=$model --resume=$resume --batch-size=$bs --device=cuda --download  