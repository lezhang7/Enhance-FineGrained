import torch
import os, json
import random
from tqdm import tqdm

from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

from read_foil_dataset import read_foils

visual7w_dir="/home/mila/l/le.zhang/scratch/datasets/visual7w/"
coco2017_dir="/home/mila/l/le.zhang/scratch/datasets/coco/2017/val2017/"
coco2014_dir="/home/mila/l/le.zhang/scratch/datasets/coco/2014/val2014/"
swig_dir="/home/mila/l/le.zhang/scratch/datasets/SWiG/"
visDial_dir="/home/mila/l/le.zhang/scratch/datasets/VisualDialog_val2018"
# ARR version
DATA = {
    "existence": [visual7w_dir,
                  'data/existence.json'],
    "plurals": [coco2017_dir,
                'data/plurals.json'],
    "counting_hard": [visual7w_dir,
                      'data/counting_hard.json'],
    "counting_small": [visual7w_dir,
                       'data/counting-small-quant.json'],
    "counting_adversarial": [visual7w_dir,
                             'data/counting_adversarial.json'],
    "relations": [coco2017_dir,
                  'data/relation.json'],
    "action replace": [swig_dir,
                       'data/action-replacement.json'],
    "actant swap": [swig_dir,
                    'data/actions/actant-swap.json'],
    "coref": [visDial_dir,
              'data/coreference-standard.json'],
    "coref_hard": [visDial_dir,
                   'data/coreference-hard.json'],
    "foil_it": [coco2014_dir,
                "data/foil-it.json"],
}


device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for instrument, foil_info in DATA.items():
    images_path = foil_info[0]
    foils_path = foil_info[1]

    foils_data = read_foils(foils_path)

    count, foil_accuracy, capt_fits, foil_detected, pairwise_acc = 0, 0, 0, 0, 0

    for foil_id, foil in tqdm(foils_data.items()):
        caption_fits = foil['mturk']['caption']

        if caption_fits >= 2:  # valid examples only (validated by mturkers)

            test_img_path = os.path.join(images_path, foil["image_file"])
            test_sentences = [foil["caption"][0], foil["foil"]]

            # run frcnn
            
            images, sizes, scales_yx = image_preprocess(test_img_path)
            output_dict = frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=frcnn_cfg.max_detections,
                return_tensors="pt"
            )

            # Very important that the boxes are normalized
            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")

            # run lxmert
            # test_sentence = [test_sentence]

            inputs = lxmert_tokenizer(
                test_sentences,
                padding="max_length",
                max_length=30,  # 20
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            output_lxmert = lxmert_base(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=inputs.token_type_ids,
                return_dict=True,
                output_attentions=False,
            )

            m = torch.nn.Softmax(dim=1)
            output = m(output_lxmert['cross_relationship_score'])
            cross_score = output_lxmert['cross_relationship_score']

            foil['lxmert'] = {'caption': 0, 'foil': 0} # 0 is not detected, 1 is detected
            foil['lxmert']['caption'] = output[0, 1].item() # probability of fitting should be close to 1 for captions
            foil['lxmert']['foil'] = output[1, 0].item() # probability of fitting, should be close to 0 for foils

            if cross_score[1, 0] == cross_score[1, 1]:  # then something is wrong with the tokenisation
                print(cross_score, test_sentences, inputs.input_ids)
            else:
                if cross_score[0, 0] < cross_score[0, 1]:  # the caption fits the image well
                    foil_accuracy += 1
                    capt_fits += 1
                if cross_score[1, 0] >= cross_score[1, 1]:
                    foil_detected += 1
                    foil_accuracy += 1
                if cross_score[0, 1] > cross_score[1, 1]:
                    pairwise_acc += 1

                count += 1

    print(f"""{instrument} sample {count}/{len(foils_data)}.
    FOIL det accuracy (acc): {foil_accuracy/count*50:.2f},
    Caption fits p_c: {capt_fits/count*100:.2f},
    FOIL detected p_f: {foil_detected/count*100:.2f},
    Pairwise accuracy acc_r: {pairwise_acc/count*100:.2f}"""
          )

    core = foils_path.split('/')[-1]
    with open(f'lxmert_results_json/{core}', 'w') as outfile:
        json.dump(foils_data, outfile)
