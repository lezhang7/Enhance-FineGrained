#!/bin/bash
echo "Downloading COCO dataset"

if [ ! -d "train2014" ]; then
    echo "Downloading train dataset from COCO"
    wget http://images.cocodataset.org/zips/train2014.zip
    unzip train2014.zip
else
    echo "train2014 directory already exists, skipping download."
fi


if [ ! -d "annotations" ]; then
    echo "Downloading annotations from COCO"
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip annotations_trainval2014.zip
    rm annotations_trainval2014.zip
else
    echo "annotations directory already exists, skipping download."
fi

echo "Creating dataset"
python dataset.py --data coco_train -bs 2048
