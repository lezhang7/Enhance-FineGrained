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
import glob

COCO_DATASET_ROOT="/home/mila/l/le.zhang/scratch/dataset/coco/2014"
train_annotations_root='/home/mila/l/le.zhang/scratch/datasets/coco/2014/annotations/captions_train2014.json'
xvlm_coco_train_annotations_root='/home/mila/l/le.zhang/scratch/X-VLM/data/finetune/coco_train.json'
val_annotations_root='/home/mila/l/le.zhang/scratch/datasets/coco/2014/annotations/captions_val2014.json'
train_root="/home/mila/l/le.zhang/scratch/datasets/coco/2014/train2014/"
# test_root="./test2014/"
val_root="/home/mila/l/le.zhang/scratch/datasets/coco/2014/val2014/"


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


class ValseDataset(Dataset):
    def __init__(self,DATA,filter_num:int = 2,single_pair:bool = False):
        self.data=[]
        self.DATA=DATA
        self.single_pair=single_pair
        for instrument, foil_info in DATA.items():
            images_path = foil_info[0]
            foils_path = foil_info[1]
            foils_data = read_foils(foils_path)
            
            data=[self.add_info(foils_data[k],k,images_path) for k in foils_data]
            data=filter(lambda x:x['mturk']['caption']>=filter_num,data)
            self.data.extend(data)
        if not single_pair:
            self.data=self.generate_label(self.data)
    def generate_label(self,data):
        print('generating order label')
        order_discrimination_data=[]
        for i in data:
            j=i.copy()
            i['label']=0
            order_discrimination_data.append(i)
            true_caption=j['caption']
            j['caption']=j['foil']
            j['foil']=true_caption
            j['label']=1
            order_discrimination_data.append(j)      
        return order_discrimination_data
            
    def add_info(self,dict,idx,images_path):
        dict['idx']=idx
        image_path=os.path.join(images_path,dict['image_file'])
        dict['image_path']=image_path
        return dict  
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        example=self.data[index]
        # print(example)
        text= [example['caption'],example['foil']]
        image=Image.open(example['image_path']).convert('RGB') 
        true_caption=0 if self.single_pair else example['label']
        return {'text':text,'image':image,'true_caption_order':true_caption,'dataset':example['dataset']}


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transforms,tokenizer=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
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
    

class ConceptualCaptionDataset(Dataset):
    def __init__(self,root:str ="/network/datasets/conceptualcaptions/",split:str= "train",small_data:int=0,samples_root:str=None):
        self.root=root
        self.small_data=small_data
        if samples_root:
            if os.path.isdir(samples_root):
                data_file_splits=glob.glob(os.path.join(samples_root,'*.npy'))
                print(f'merging {len(data_file_splits)} splied files from {samples_root}')
                self.samples=[]
                for file_split in data_file_splits:
                    self.samples.extend(self.loadList(file_split))
            else:
                self.samples=self.loadList(samples_root)
        else:
            self.samples=self.load_data(root,split)
    def load_data(self,data_dir,split):
        # Loading dataset from local disk
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        assert split in ['train', 'val']
        if split == 'train':
            data_dir = data_dir / 'Train'
        elif split == 'val':
            data_dir = data_dir / 'Validation'

        captions = []
        with open(next(data_dir.glob('*.tsv'))) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                captions.append(row[0])
        samples = []
        for image_file in data_dir.glob('*/*'):
            id = int(image_file.name.split('-')[0])
            caption = captions[id]
            samples.append({'idx':id, 'image_path':str(image_file),"caption":caption})
            if self.small_data and len(samples)==self.small_data:
                break
        return samples
    def read_img(self,file_path):
        return Image.open(file_path).convert("RGB")
    def loadList(self,file_path):
        # the filename should mention the extension '.npy'
        tempNumpyArray=np.load(file_path,allow_pickle=True)
        return tempNumpyArray.tolist()
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        out=self.samples[index]
        if 'image' not in out:
            out.update({'image':self.read_img(out['image_path'])})
        return out

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
        print(f"POS compledted, cost {m_time-s_time} s")
        dataset=list(map(self.mask_captions,zip(docs,dataset)))
        e_time=time.time()
        print(f'POS and MASK completed! cost {e_time-s_time} s')
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
    elif args.data=='coco_val':
        dataset=CocoDataset(os.path.join(COCO_DATASET_ROOT,val_root),val_annotations_root,None)
        samples=list(dataset.dataset.anns.values())
    elif args.data=='coco_xvlm':
        samples=json.load(open(xvlm_coco_train_annotations_root, 'r'))
    DA=TextAugment()
    os.makedirs(f"processed_dataset/{args.data}",exist_ok=True)
    for split_idx,split_star_index in enumerate(range(0,len(samples),args.split_num)):
        data=samples[split_star_index:split_star_index+args.split_num]
        save_path=os.path.join(f'processed_dataset/{args.data}/processed_dataset{split_idx}.npy')
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
    
        