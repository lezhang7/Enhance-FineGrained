import argparse
import os
import pandas as pd
import re
from torch.utils.data import DataLoader
import json
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
import numpy as np
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--dataset", default="VG_Relation", type=str, choices=["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"])
    parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
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
def check_json(json_path, key:str ):
    # if exists return True
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if key in json_data.keys():
            return True 
    return False
    
def main(args):

   
    if args.resume and args.resume.endswith(".pt"):
        match = re.search(r'([^/]+)\.pt$', args.resume)
        resume=f"{args.model_name}_{match.group(1)}"
    else:
        resume=args.model_name
    result_path=os.path.join("./", f"{args.dataset}.json")
    result={}
    if check_json(result_path,resume):
        raise ValueError(f"resume: {resume} already exists") 
    else:
        print(f"evaluating resume: {resume} on {args.dataset}")
    
    seed_all(args.seed)
    
   
    # path
   
    output_file=os.path.join("./outputs/"+resume)
    print(f"Saving results to {output_file}")
   

    # calculating scores
    model, image_preprocess = get_model(args.model_name, args.device,args.resume)
    
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    
   
    collate_fn = _default_collate if image_preprocess is None else None
    
    joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    scores = model.get_retrieval_scores_batched(joint_loader)
    # save scores 
    os.makedirs(os.path.join(output_file),exist_ok=True)
    np.save(os.path.join(output_file,f"{args.dataset}.npy"),scores)
    result_records = dataset.evaluate_scores(scores)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Seed": args.seed})
    df = pd.DataFrame(result_records)
    
    if args.dataset=="VG_Relation":
        symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
        df = df[~df.Relation.isin(symmetric)]  
    if args.save_scores:
        save_scores(scores, args)

    # compute and save results
    acc=0
    acc=df['Accuracy'].mean()
    print(f"{args.dataset} acc: {acc}")
 
    result={resume:acc}
    update_json(result_path,result)

    # os.makedirs(output_file,exist_ok=True)
    # df.to_csv(os.path.join(output_file,f"{args.dataset}.csv"))


    
if __name__ == "__main__":
    args = config()
    if args.model_name=='ours':
        assert args.resume is not None, "Test ours model must assign checkpont path"
    main(args)