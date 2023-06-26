import torch
from model_zoo import get_model
import os, json
from PIL import Image
import random
from tqdm import tqdm
from read_foil_dataset import read_foils
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
import argparse
import itertools

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    

CATEGORIES={
    "Existence":["existence"],
    "Plurality":["plurals"],
    "Counting":["counting_hard","counting_small","counting_adversarial"],
    "Relations":["relations"],
    "Actions":["action replace","actant swap"],
    "Coreference":["coref","coref_hard"],
    "Foil it":["foil_it"],
}
SUB_CATEGORIES=list(itertools.chain.from_iterable(CATEGORIES.values()))


class VASELDATASET(Dataset):
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

    def __init__(self,catergory,image_preprocess):

        self.images_path,self.foils_path = self.DATA[catergory]
        self.image_preprocess=image_preprocess
        self.foils_data = list(read_foils(self.foils_path).items())

    def __getitem__(self, index):

        foil_id, foil = self.foils_data[index]    
        if "COCO_val2014" in foil["image_file"]:
            images_path=self.coco2014val_dir
        elif "COCO_train2014" in foil["image_file"]:
            images_path=self.coco2014train_dir
        elif "COCO_val2017" in foil["image_file"]:
            images_path=self.coco2017val_dir
        elif "COCO_train2017" in foil["image_file"]:
            images_path=self.coco2017train_dir
        else:
            images_path=self.images_path
        test_img_path = os.path.join(images_path, foil["image_file"])
        image = self.image_preprocess(Image.open(os.path.join(test_img_path)).convert('RGB'))
        item={"image_options": [image], "caption_options": [foil["caption"],foil["foil"]]}
        return item
    
    def __len__(self):
        return len(self.foils_data)
    
    def read_foils(self,foils_path):
        if "original-foil-dataset" in foils_path:
            foils_data = self.read_foil_dataset(foils_path)
        else:
            with open(foils_path) as json_file:
                foils_data = json.load(json_file)
        foils_data={foil_id:foil for foil_id, foil in foils_data.items() if foil['mturk']['caption']>2 }
        return foils_data

    def read_foil_dataset(self,foils_path):
        """
        Read in the data of the original foil dataset and convert it on the fly to our format (dict/json).
        """
        with open(foils_path) as json_file:
            foil_dataset = json.load(json_file)

        foils_data = {}  # our format

        for foil in foil_dataset['annotations']:
            # For unimodal models, we always need foil, non-foil pairs to compare perplexity.
            if foil['foil'] == True:  # we have a foil not foil pair
                # recover the original sentence
                orig_sentence = foil['caption'].replace(foil['foil_word'], foil['target_word'])
                image_id = foil['image_id']
                foils_data[foil["foil_id"]] = {'dataset': 'FOIL dataset',
                                            'dataset_idx': foil["foil_id"],
                                            'original_split': 'test',
                                            'linguistic_phenomena': 'noun phrases',
                                            'image_file': f'COCO_val2014_{str(image_id).zfill(12)}.jpg', # COCO_val2014_000000522703.jpg all are "val"
                                            'caption': orig_sentence,
                                            'foils': [foil['caption']],
                                            'classes': foil['target_word'],
                                            'classes_foil': foil['foil_word'],
                                            }

        return foils_data
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            result_records.append({
                "Relation": relation,
                "Accuracy": correct_mask[relation_mask].mean(),
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })
        return result_records
# parse arguments 

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default='xvlm-coco', type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--pretrained", default=None, type=str)
    parser.add_argument("--output-dir", default="./output/", type=str)
    parser.add_argument("--resume", default=None, type=str,help="path to .pt checkpoint file")
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
def check_json(json_path, key:str ,epoch_num:str):
    # if exists return True
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if key in json_data.keys():
            if epoch_num in json_data[key].keys():
                return True 
    return False
def main(args):
    os.makedirs(args.output_dir,exist_ok=True)

    # if args.pretrained is not None and "checkpoints" in args.pretrained:
    #     checkpoint=re.findall(r'Outputs/(.*?)/checkpoints',args.pretrained)[0]
    #     epoch_num=re.findall(r"(epoch.*).pt",args.pretrained)[0]
    #     result_key=f"{checkpoint}"
    # else:
    #     result_key=f"{args.model_name}"

    if args.resume:
        if args.model_name=='ours':
            resume=re.findall(r'Outputs/(.*?)/checkpoints',args.resume)[0]
        elif args.model_name=='xvlm-coco':
            resume=re.findall(r'.cache/(.*?)/checkpoint',args.resume)[0]
        epoch_num=re.findall(r"(epoch.*).pt",args.resume)[0]    
    else:
        resume=args.model_name
        epoch_num="pretrained"

    output_dir=os.path.join(args.output_dir,"result_xvlm.json")
    if check_json(output_dir,resume,epoch_num):
        raise ValueError(f"result_key: {resume}_{epoch_num} already exists") 
    else:
        print(f"evaluating result_key: {resume}_{epoch_num}")

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, image_preprocess = get_model(args.model_name, device,args.resume)


    overall_results={}


    # path
   
    for subcategory in SUB_CATEGORIES:
        dataset=VASELDATASET(subcategory,image_preprocess)
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)       
        scores = model.get_retrieval_scores_batched(dataloader)
        scores_i2t = scores[1]
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        correct_mask=correct_mask.mean()
        overall_results[subcategory]=round(correct_mask*100,2)
        print(f"{subcategory} Acc: {overall_results[subcategory]}")

    overall_results['avg']=round(np.array(list(overall_results.values())).mean(),2)
    category_results=[]
    for category,splits in CATEGORIES.items():
        acc=0
        for i in splits:
            acc+=overall_results[i]
        category_results.append(str(round(acc/len(splits),1)))
    category_results.append(str(overall_results['avg']))
    overall_results['latex_results']='&'.join(category_results)
    result={resume:{epoch_num:overall_results}}
    update_json(output_dir,result)
    

if __name__=="__main__":
  
    
    args=arg_parse()
    main(args)


