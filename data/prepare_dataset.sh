echo "Downloading COCO dataset"
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/train2014.zip
unzip annotations_trainval2014.zip
unzip train2014.zip
rm annotations_trainval2014.zip
rm train2014.zip

echo "Creating dataset"
python dataset.py --data coco_train
