import torch
# from processing_image import Preprocess
# from modeling_frcnn import GeneralizedRCNN
# from utils import Config
# from transformers import LxmertTokenizer, LxmertForPreTraining
import os, json
from PIL import Image
import random
from tqdm import tqdm
import open_clip
from read_foil_dataset import read_foils
import numpy as np
import re
import argparse

# replace these dataset paths with your own
visual7w_dir="/home/mila/l/le.zhang/scratch/datasets/visual7w/"
coco2017val_dir="/home/mila/l/le.zhang/scratch/datasets/coco/2017/val2017/"
coco2014val_dir="/home/mila/l/le.zhang/scratch/datasets/coco/2014/val2014/"
coco2014train_dir="/home/mila/l/le.zhang/scratch/datasets/coco/2014/train2014/"
coco2017train_dir="/home/mila/l/le.zhang/scratch/datasets/coco/2017/train2017/"
swig_dir="/home/mila/l/le.zhang/scratch/datasets/SWiG/"
visDial_dir="/home/mila/l/le.zhang/scratch/datasets/VisualDialog_val2018"


# ARR version
DATA = {
    "existence": [visual7w_dir,
                  './data/existence.json'],
    "plurals": [coco2017val_dir,
                './data/plurals.json'],
    "counting_hard": [visual7w_dir,
                      './data/counting-hard.json'],
    "counting_small": [visual7w_dir,
                       './data/counting-small-quant.json'],
    "counting_adversarial": [visual7w_dir,
                             './data/counting-adversarial.json'],
    "relations": [coco2017val_dir,
                  './data/relations.json'],
    "action replace": [swig_dir,
                       './data/action-replacement.json'],
    "actant swap": [swig_dir,
                    './data/actant-swap.json'],
    "coref": [coco2014train_dir,
              './data/coreference-standard.json'],
    "coref_hard": [visDial_dir,
                   './data/coreference-hard.json'],
    "foil_it": [coco2014val_dir,
                "./data/foil-it.json"],
}
CATEGORIES={
    "Existence":["existence"],
    "Plurality":["plurals"],
    "Counting":["counting_hard","counting_small","counting_adversarial"],
    "Relations":["relations"],
    "Actions":["action replace","actant swap"],
    "Coreference":["coref","coref_hard"],
    "Foil it":["foil_it"],
    "avg":["avg"]
}
# parse arguments 

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="'ViT-B-32'", type=str)
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--output-dir", default="./output/", type=str)
    return parser.parse_args()

def merge_dict(dict1,dict2):
    for key in dict2.keys():
        if key in dict1.keys():
            dict1[key].update(dict2[key])
        else:
            dict1[key]=dict2[key]
    return dict1
def update_json(json_path, value:dict):
    # check if json_path exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {}
    json_data=merge_dict(json_data,value)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=6)
# check if json file has key
def check_json(json_path, key:str ):
    # if exists return True
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if key in json_data.keys():
            return True 
    return False
def main(args):
    os.makedirs(args.output_dir,exist_ok=True)

    result_key=f"{args.model_name}_{args.pretrained}"
    
    

    output_dir=os.path.join(args.output_dir,"result_clip.json")
    if check_json(output_dir,result_key):
        raise ValueError(f"result_key: {result_key} already exists") 
    else:
        print(f"evaluating result_key: {result_key}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    overall_results={}
    overall_results={}


    # path
   

    for instrument, foil_info in DATA.items():
        images_path = foil_info[0]
        foils_path = foil_info[1]
        foils_data = read_foils(foils_path)
        
        correct=0
        valid_num=0
        for foil_id, foil in tqdm(foils_data.items(),desc=instrument):
            caption_fits = foil['mturk']['caption']

            if caption_fits >= 2:  # valid examples only (validated by mturkers)
                valid_num+=1
                if "COCO_val2014" in foil["image_file"]:
                    images_path=coco2014val_dir
                elif "COCO_train2014" in foil["image_file"]:
                    images_path=coco2014train_dir
                elif "COCO_val2017" in foil["image_file"]:
                    images_path=coco2017val_dir
                elif "COCO_train2017" in foil["image_file"]:
                    images_path=coco2017train_dir
            
                test_img_path = os.path.join(images_path, foil["image_file"])
                test_sentences = tokenizer([foil["caption"], foil["foil"]]).to(device)
                image = image_preprocess(Image.open(os.path.join(test_img_path)).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(test_sentences)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                correct += logits_per_image.softmax(dim=1).cpu()[0][0]>0.5
        acc=(correct/valid_num).item()*100
        overall_results[instrument]=acc
        print(f"{instrument} Acc: {acc:.2f}")

    overall_results['avg']=np.array(list(overall_results.values())).mean()
    category_results=[]
    for category,splits in CATEGORIES.items():
        acc=0
        for i in splits:
            acc+=overall_results[i]
        category_results.append(str(round(acc/len(splits),1)))
    category_results.append(str(round(overall_results['avg'],1)))
    overall_results['latex_results']='&'.join(category_results)
    result={result_key:overall_results}
    update_json(output_dir,result)
    
# define update json function

if __name__=="__main__":
    args=arg_parse()
    main(args)


