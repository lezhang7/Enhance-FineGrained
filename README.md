# [Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Compositional Understanding](https://arxiv.org/abs/2306.08832)

:tada: <span style="color: red;">**The paper was accepted to CVPR 2024**</span>:

TL;DR: We Propose **two losses** on our **generated hard negative** examples to enhance model's **compositional understanding** ability for CLIP.

![motivation](/assets/motivation.png)

**This repo forks from wonderful [OpenCLIP](https://github.com/mlfoundations/open_clip)**, for model and training details, please refer to original repo.

# :ballot_box_with_check: Checkpoints

The checkpoints could be downloaded directly using gdown with following script:

``````bash
pip install --upgrade --no-cache-dir gdown # must update gdown to avoid bugs, thanks to https://github.com/wkentaro/gdown/issues/146
gdown 1DWPw3CtGh5cHz9bW_-iXRSG7BBUVl13K #download checkpoint for CE-CLIP
``````

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
bash run_multiple_nodes.sh
```

The result checkpoint will be at `Enhance-FineGrained/src/Outputs`

# Evaluation

We evaluate our method on four downstream task [ARO](https://github.com/mertyg/vision-language-models-are-bows), [VALSE](https://github.com/Heidelberg-NLP/VALSE) and [VL-CheckList](https://github.com/om-ai-lab/VL-CheckList), and very recent [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) and we also provide evaluation code. However, one need go to official github page to download dataset to evaluate on them.

### ARO&VALSE

![ARO](/assets/aro.png)

Evaluation code for ARO is included in `Enhance-FineGrained/vision-language-models-are-bows`, to reproduce results, you need 

1. set up environment by running `bash Enhance-FineGrained/vision-language-models-are-bows/scripts/create_environment.sh`

2. `cd Enhance-FineGrained/vision-language-models-are-bows/scripts` and change the checkpoint path in `reproduce_aro.sh`, then **run the script to reproduce the results**. *Note that dataset will be download automatically*

   ---

3. Evaluation code for VALSE is included in `Enhance-FineGrained/VALSE`, to reproduce results on valse, please download dataset [here](https://github.com/Heidelberg-NLP/VALSE) first. **Then replace dataset** path in `Enhance-FineGrained/VALSE/clip_valse_eval.py` `Enhance-FineGrained/VALSE/xvlm_valse_eval.py`

4. replace `$checkpoint` in `Enhance-FineGrained/VALSE/scripts` then run the scripts, evaluation results will be included in `/home/mila/l/le.zhang/scratch/Enhance-FineGrained/VALSE/output`

### VL-CheckList [Not Suggested]

![vlchecklist](/assets/vlchecklist.png)

:exclamation: **Note: The original dataset is not complete, we encourage skip this dataset** 

Please refer to [official github](https://github.com/om-ai-lab/VL-CheckList) repo to download dataset and perform evaluation. *Note that Downloading the dataset can be quite cumbersome*

we provide script at [here](https://github.com/rabiulcste/vl_checklist/tree/ca0c68d1f457f670139feb75a6b884adff88aeee)

### :star2: SugarCrepe

![sugarcrepe](/assets/sugarcrepe.png)

[SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) is a benchmark for faithful vision-language compositionality evaluation. This dataset **fix a several biases** in all above benchmarks *rendering them hackable that blind models with no access to the image outperform state-of-the-art vision-language models*. 

to evaluate on this dataset, simply clone their repo and follow their installation setup, and assign retrained to our checkpoints

```python
python main_eval.py --model ViT-B-32 --pretrained Enhance-FineGrained/clip/epoch_5.pt \
    --output ./output \
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/ \
```

### Ablations

Our method entails curriculum learning, which is validated by the growth of adaptive threshold

![abaltion](/assets/abaltion.png)

# :paperclip: Citation

``````bibtex
@article{zhang2023contrasting,
  title={Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Fine-grained Understanding},
  author={Zhang, Le and Awal, Rabiul and Agrawal, Aishwarya},
  journal={arXiv preprint arXiv:2306.08832},
  year={2023}
}
``````



# :email: Contact

please let us know if you have further questions or comments, reach out to [le.zhang@mila.quebec](mailto:le.zhang@mila.quebec)
