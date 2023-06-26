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

import numpy as np
import re
import argparse

import random
from training.data import NpyDataset
from torch.utils.data import DataLoader


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
def check_json(json_path, key:str):
    # if exists return True
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if key in json_data.keys():
            return True 
    return False

def compute_similarity(logit_scale,image_features, text_features, bs = 1000):
    # compute similarity
    dim0 = image_features.shape[0]
    dim1 = text_features.shape[0]
    similarity_scores = torch.zeros(dim0, dim1)
    for v in range(0, dim0, bs):
        for t in range(0, dim1, bs):
            batch_visual_emb = image_features[v:v+bs].cuda()
            batch_caption_emb = text_features[t:t+bs].cuda()
            with torch.no_grad():
                logits = logit_scale.cuda() * batch_visual_emb @ batch_caption_emb.t()
                logits.cpu()
            similarity_scores[v:v+bs,t:t+bs] = logits
            torch.cuda.empty_cache()
            del batch_visual_emb, batch_caption_emb, logits

    print('Done similarity')
    return similarity_scores


def get_itr_scores(logit_scale, image_features, text_features):
    bs=2000
    split_num=50
    image_len = image_features.shape[0]
    text_len  = text_features.shape[0]
    split_size=image_len//split_num
    total_ranks=[]

    for eid,partial_index in tqdm(enumerate(range(0, image_len, split_size))):
        partial_image_features = image_features[partial_index:partial_index+split_size]
        max_pairs=partial_image_features.shape[0]
        sub_similarity_scores = torch.zeros((max_pairs, text_len))
        for v in range(0, max_pairs, bs):
            for t in range(0, text_len, bs):
                batch_visual_emb = partial_image_features[v:v+bs]
                batch_caption_emb = text_features[t:t+bs]
                with torch.no_grad():
                    logits = logit_scale * (batch_visual_emb @ batch_caption_emb.t())
                sub_similarity_scores[v:v+bs,t:t+bs] = logits.detach().cpu()
        a2b_sims=sub_similarity_scores.detach().cpu()
        ground_truth=torch.arange(eid*split_size,min((eid+1)*split_size,image_len)).reshape(-1,1)
        ranking = torch.argsort(a2b_sims, descending=True) 
        preds = torch.where(ranking == ground_truth)[1]
        total_ranks.append(preds)

    metrics={}
    total_ranks=torch.cat(total_ranks)
    def mean_of_list(l):
        return round((sum(l)/len(l)).item(),5)
    
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = mean_of_list(total_ranks < k)
    metrics[f"mean_rank"] = round(total_ranks.mean(dtype=torch.float32).item())+ 1
    metrics[f"median_rank"] = np.floor(np.median(total_ranks)) + 1
    return metrics
def eval_clip(args):
    
    # path
    epoch_num=None
    if "checkpoints" in args.pretrained :
        pretrained_name=re.findall(r'Outputs/(.*?)/checkpoints',args.pretrained)[0]
        epoch_num=re.findall(r"(epoch.*).pt",args.pretrained)[0]
    else:
        pretrained_name=args.pretrained
    if not epoch_num:
        epoch_num="pretrained"
    checkpoint_key=f"{pretrained_name}_{epoch_num}"

    # check if result_key exists
    json_output_dir=os.path.join(args.output_dir,f"{'test' if args.test else 'hnitr_clip'}.json")
    
    if check_json(json_output_dir,checkpoint_key):
        raise ValueError(f"result_key: {checkpoint_key} already exists") 
    else:
        print(f"evaluating result_key: {checkpoint_key}")
    print("output_dir:",json_output_dir)
    # loading model
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    # loading dataset
    eval_dataset=NpyDataset(args.dataset,image_preprocess,args.size,tokenizer)
    eval_dataloader=DataLoader(eval_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    print(f"toatal dataset size: {len(eval_dataset)}")
    
    # extract features
    gt_caption_features=[]
    hn_caption_features=[]
    all_image_features=[]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(eval_dataloader,desc="extracting features"):
            gt_caption=i[1][:,0,:].to(device)
            hn_caption=i[1][:,1:,:].reshape(-1,77).to(device)
            images=i[0].to(device)
            image_features = model.encode_image(images)
            gt_text_features = model.encode_text(gt_caption)
            hn_text_features = model.encode_text(hn_caption)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            gt_text_features /= gt_text_features.norm(dim=-1, keepdim=True)
            hn_text_features /= hn_text_features.norm(dim=-1, keepdim=True)
            gt_caption_features.append(gt_text_features)
            hn_caption_features.append(hn_text_features)
            all_image_features.append(image_features)
    all_caption_features=torch.cat((torch.cat(gt_caption_features),torch.cat(hn_caption_features)))
    all_image_features=torch.cat(all_image_features)
    #save features as binary file
    features={"caption_features":all_caption_features,"image_features":all_image_features}
    torch.save(features,os.path.join(args.output_dir,"saved_features",f"{checkpoint_key}{'test' if args.test else ''}_features.pt"))

    metric=get_itr_scores(model.logit_scale,all_image_features,all_caption_features)
    result={checkpoint_key:metric}
    # save scores
    print(f" update json file: {json_output_dir}")
    update_json(json_output_dir,result)




def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="'ViT-B-32'", type=str)
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--output-dir", default="./itr_results/", type=str)
    parser.add_argument("--test",action="store_true")   
    
    args = parser.parse_args()

    if args.pretrained!="openai" and not os.path.exists(args.pretrained):
        raise ValueError(f"pretrained file {args.pretrained} does not exist")
    print("---- arguments ----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-------------------")

    return args

if __name__=="__main__":
    args=arg_parse()
    eval_clip(args)


