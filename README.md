# Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Fine-grained Understanding

Training code and training data augmentation code for the paper "[Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Fine-grained Understanding](https://arxiv.org/abs/2306.08832)" 

TL;DR: We Propose **two losses** to enhance model's **fine-grained understanding** ability for any model with image-text contrastive loss like CLIP. The two losses are applied on our **generated hard negative** examples.

<img src="/Users/zhangle/Desktop/conference/neurips figures/overview.png" alt="overview" style="zoom:22%;" />

**This repo forks from [OpenCLIP](https://github.com/mlfoundations/open_clip)**, for model and training details, please refer to original repo.

# Training

### 1. Generating Training dataset

The training data is generated based on COCO 2014, so you can either [download](https://cocodataset.org/#download) by yourself and assign coco `dataset_path` in `dataset.py` or **you can simply run following script to download and generate** dataset

``````python
cd data/
bash prepare_dataset.sh
``````

### 2. Training 

you need to specify training parameters in scrips/run_all.sh such as  `--gres=gpu:a100:2` and `batch_size`, please refer to this script file to see more details, to simply run the training, using following scritps

```python
cd scripts/
bash run_all.sh
```

