# Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Fine-grained Understanding

Training code and training data augmentation code for the paper "[Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Fine-grained Understanding](https://arxiv.org/abs/2306.08832)" 

TL;DR: We Propose **two losses** to enhance model's **fine-grained understanding** ability for any model with image-text contrastive loss like CLIP. The two losses are applied on our **generated hard negative** examples.

<img src="https://p.ipic.vip/2vo9it.png" alt="overview" style="zoom:22%;" />

**This repo forks from [OpenCLIP](https://github.com/mlfoundations/open_clip)**, for model and training details, please refer to original repo.

# :ballot_box_with_check: Checkpoints

**We release both clip-enhanced and xvlm-enhanced checkpoints at [here](https://drive.google.com/drive/folders/1rpt_YpqSatuWTUDT9uMXkU1RUSBfWec1?usp=sharing)**

# Training

The two losses are included in `Enhance-FineGrained/src/open_clip/loss.py` `Clip_DALoss`, the training file is at `Enhance-FineGrained/src/training/train.py`. Here are scripts to reproduce the results.

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

The result checkpoint will be at `Enhance-FineGrained/src/Outputs`



# Evaluation

We evaluate our method on four downstream task [ARO](https://github.com/mertyg/vision-language-models-are-bows), [VALSE](https://github.com/Heidelberg-NLP/VALSE) and [VL-CheckList](https://github.com/om-ai-lab/VL-CheckList), and very recent [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) and we also provide evaluation code. However, one need go to official github page to download dataset to evaluate on them.

### ARO

![Screenshot 2023-07-01 at 8.40.22 PM](https://p.ipic.vip/jjzo57.png)

Evaluation code for ARO is included in `Enhance-FineGrained/vision-language-models-are-bows`, to reproduce results, you need 

1. set up environment by running `bash Enhance-FineGrained/vision-language-models-are-bows/scripts/create_environment.sh`
2. `cd Enhance-FineGrained/vision-language-models-are-bows/scripts` and change the checkpoint path in `reproduce_aro.sh`, then **run the script to reproduce the results**. *Note that dataset will be download automatically*

### VALSE

1. Evaluation code for VALSE is included in `Enhance-FineGrained/VALSE`, to reproduce results on valse, please download dataset [here](https://github.com/Heidelberg-NLP/VALSE) first. **Then replace dataset** path in `Enhance-FineGrained/VALSE/clip_valse_eval.py` `Enhance-FineGrained/VALSE/xvlm_valse_eval.py`
2. replace `$checkpoint` in `Enhance-FineGrained/VALSE/scripts` then run the scripts, evaluation results will be included in `/home/mila/l/le.zhang/scratch/Enhance-FineGrained/VALSE/output`

### VL-CheckList

<img src="https://p.ipic.vip/dm5adr.png" alt="Screenshot 2023-07-01 at 8.40.28 PM" style="zoom:50%;" />Please refer to [official github](https://github.com/om-ai-lab/VL-CheckList) repo to download dataset and perform evaluation. *Note that Downloading the dataset can be quite cumbersome*

### :star2: SugarCrepe

<img src="https://p.ipic.vip/4ba1ok.png" alt="Screenshot 2023-07-01 at 8.40.32 PM" style="zoom:50%;" />

[SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) is a benchmark for faithful vision-language compositionality evaluation. This dataset **fix a several biases** in all above benchmarks *rendering them hackable that blind models with no access to the image outperform state-of-the-art vision-language models*. 

to evaluate on this dataset, simply clone their repo and follow their installation setup, and assign retrained to our checkpoints

```python
python main_eval.py --model ViT-B-32 --pretrained Enhance-FineGrained/clip/epoch_5.pt \
    --output ./output \
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```



# :paperclip: Citation

``````bibtex
@misc{zhang2023contrasting,
      title={Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Fine-grained Understanding}, 
      author={Le Zhang and Rabiul Awal and Aishwarya Agrawal},
      year={2023},
      eprint={2306.08832},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``````



# :email: Contact

please let us know if you have further questions or comments, reach out to [le.zhang@mila.quebec](mailto:le.zhang@mila.quebec)
