import json
import os
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import csv
import random
from transformers import pipeline
import spacy
from typing import List, Set, Dict, Tuple
import numpy as np
import argparse
import time
from pycocotools.coco import COCO

COCO_DATASET_ROOT="./"
train_annotations_root='annotations/captions_train2014.json'
xvlm_coco_train_annotations_root='/home/mila/l/le.zhang/scratch/X-VLM/data/finetune/coco_train.json'
val_annotations_root='annotations/captions_val2014.json'
train_root="train2014/"
# test_root="./test2014/"
val_root="val2014/"


def read_foils(foils_path):
    if "original-foil-dataset" in foils_path:
        foils_data = read_foil_dataset(foils_path)
    else:
        with open(foils_path) as json_file:
            foils_data = json.load(json_file)
    return foils_data

def read_foil_dataset(foils_path):
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


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, image_root, json, transforms,tokenizer=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = image_root
        self.dataset = COCO(json)
        self.ids = list(self.dataset.anns.keys())
        self.transforms = transforms
        self.tokenize = tokenizer

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        dataset = self.dataset
        ann_id = self.ids[index]
        caption = dataset.anns[ann_id]['caption']
        img_id = dataset.anns[ann_id]['image_id']
        path = dataset.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        caption=self.tokenize(caption)
        return image, caption

    def __len__(self):
        return len(self.ids)
    

class MaskedCaptionsDataset(Dataset):
    def __init__(self,masked_captions):
        self.masked_captions=masked_captions
    def __getitem__(self, index):
        return self.masked_captions[index]
    def __len__(self):
        return len(self.masked_captions)
    
class TextAugment(object):
    def __init__(self):
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.unmasker = pipeline('fill-mask', model='distilroberta-base',device=self.device,top_k=5)
        self.nlp = spacy.load("en_core_web_sm",exclude=['ner','parser'])
    def mask_captions(self,data):
        doc,data_item=data[0],data[1]
        org_text=doc.text
        nouns=[]
        adjactive=[]
        verbs=[]
        for token in doc:
            if token.pos_=='NOUN':
                nouns.append(token.text)
            elif token.pos_=='ADJ':
                adjactive.append(token.text)
            elif token.pos_=='VERB':
                verbs.append(token.text)
        if len(nouns)>2:
            text=org_text.replace(nouns[0],'temp1')
            text=text.replace(nouns[1],nouns[0])
            text=text.replace('temp1',nouns[1])
            # print(f"{'relation change result':<25}:{text}")
            data_item.update({'relation_aug_caption':text})
        else:
            data_item.update({'relation_aug_caption':'###'})
        if adjactive:
            data_item.update({'adj_aug_caption':org_text.replace(random.choice(adjactive),'<mask>',1)})
        else:
            data_item.update({'adj_aug_caption':'<mask>'})
        if nouns:
            data_item.update({'noun_aug_caption':org_text.replace(random.choice(nouns),'<mask>',1)})
        else:
            data_item.update({'noun_aug_caption':'<mask>'})
        if verbs:
            data_item.update({'verb_aug_caption':org_text.replace(random.choice(verbs),'<mask>',1)})
        else:
            data_item.update({'verb_aug_caption':'<mask>'})
        return data_item
    def select_unmasked_captions(self,unmasked_resutls):
        return unmasked_resutls[min(3,len(unmasked_resutls))]['sequence']
    def loadList(self,filename):
        # the filename should mention the extension 'npy'
        tempNumpyArray=np.load(filename,allow_pickle=True)
        return tempNumpyArray.tolist()
    def __call__(self,dataset,save_name:str=None):
        print(f'Beging POS and MASK, total data length {len(dataset)}')
        s_time=time.time()
        docs=list(self.nlp.pipe([data_item['caption'] for data_item in dataset],n_process=8))
        m_time=time.time()
        print(f"POS compledted, cost {m_time-s_time:.2f} s")
        dataset=list(map(self.mask_captions,zip(docs,dataset)))
        e_time=time.time()
        print(f'POS and MASK completed! cost {e_time-s_time:.2f} s')
        masked_captions=[]
        for data_item in dataset:
            for key,value in data_item.items():
                if isinstance(value,str) and '<mask>' in value:
                    masked_captions.append(value)
        masked_captions_dataset=MaskedCaptionsDataset(masked_captions)
        maksed_results=[]
        print('utilizing pretrained model to fill the mask')
        with torch.no_grad():
            for out in tqdm(self.unmasker(masked_captions_dataset,batch_size=512),total=len(masked_captions_dataset)):
                # maksed_results.append(unmasker(masked_captions,batch_size=2048))
                maksed_results.append(out)

        torch.cuda.empty_cache()
        unmaked_captions=list(map(self.select_unmasked_captions,maksed_results))
        for idx,unmased_caption in zip(range(len(dataset)),[unmaked_captions[i:i+3] for i in range(0,len(unmaked_captions),3)]):
            # print(idx,unmased_caption)
            dataset[idx]['adj_aug_caption']=unmased_caption[0]
            dataset[idx]['noun_aug_caption']=unmased_caption[1]
            dataset[idx]['verb_aug_caption']=unmased_caption[2]
            valid_caption=[]
            for key,item in dataset[idx].items():
                if 'caption' in key:
                    if item!='###':
                        valid_caption.append(1)
                    else:
                        valid_caption.append(0)
            dataset[idx].update({"valid_caption":valid_caption})
        if save_name:
            print(f'Save processed file as {save_name}')
            with open(save_name,"wb") as f:
                np.save(f,dataset)
        return dataset
def augmentation(args):

    print('loading dataset from local folder')
    if args.data=='coco_train':
        dataset=CocoDataset(os.path.join(COCO_DATASET_ROOT,train_root),train_annotations_root,None)
        samples=list(dataset.dataset.anns.values())
    # elif args.data=='coco_val':
    #     dataset=CocoDataset(os.path.join(COCO_DATASET_ROOT,val_root),val_annotations_root,None)
    #     samples=list(dataset.dataset.anns.values())
    elif args.data=='coco_xvlm':
        samples=json.load(open(xvlm_coco_train_annotations_root, 'r'))
    DA=TextAugment()
    os.makedirs(f"generated_data/{args.data}",exist_ok=True)
    for split_idx,split_star_index in enumerate(range(0,len(samples),args.split_num)):
        data=samples[split_star_index:split_star_index+args.split_num]
        save_path=os.path.join(f'generated_data/{args.data}/processed_dataset{split_idx}.npy')
        DA(data,save_path)
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_num',type=int,default=50000)
    parser.add_argument('--data',type=str,required=True,choices=['coco_train','coco_val','coco_xvlm'])
    args = parser.parse_args()
    return args
if __name__=="__main__":
    print('perfrom data augmentation',flush=True)
    args=get_arg_parser() 
    augmentation(args)
    
        