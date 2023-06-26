import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from dataset_zoo.perturbations import get_text_perturb_fn, get_image_perturb_fn
from misc import seed_all, _default_collate, save_scores
import re
import json

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--dataset", default="COCO_Retrieval", type=str, choices=["COCO_Retrieval", "Flickr30k_Retrieval"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--text-perturb-fn", default=None, type=str, help="Perturbation function to apply to the text.")
    parser.add_argument("--image-perturb-fn", default=None, type=str, help="Perturbation function to apply to the images.")
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--resume", default=None, type=str)
    return parser.parse_args()

    
def main(args):
    seed_all(args.seed)

    # path
    epoch_num="epoch_0"
    resume=args.model_name
    if args.resume and args.model_name=='ours':
        resume=re.findall(r'Outputs/(.*?)/checkpoints',args.resume)[0]
        epoch_num=re.findall(r"(epoch.*).pt",args.resume)[0]
    output_file=os.path.join("./outputs/"+resume+"_"+epoch_num+".pt")
    print(f"Saving results to {output_file}")



    model, image_preprocess = get_model(args.model_name, args.device,args.resume)
    text_perturb_fn = get_text_perturb_fn(args.text_perturb_fn)
    image_perturb_fn = get_image_perturb_fn(args.image_perturb_fn)
    
    
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=args.download)
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
    collate_fn = _default_collate if image_preprocess is None else None
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    if text_perturb_fn is not None:
        loader.dataset.text = [text_perturb_fn(t) for t in loader.dataset.text]
    
    scores = model.get_retrieval_scores_dataset(loader)
    result_records = dataset.evaluate_scores(scores)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Text Perturbation Strategy": args.text_perturb_fn,
                       "Seed": args.seed, "Image Perturbation Strategy": args.image_perturb_fn})
    
    df = pd.DataFrame(result_records)

   
    if args.save_scores:
        save_scores(scores, args)
    # compute and save results
    result_path=os.path.join("./", f"{args.dataset}.json")
    
    result={}
    if os.path.exists(result_path):
        with open(result_path,"r") as f:
            try:
                result=json.load(f)
            except:
                pass

    model_checkpoint=resume if args.model_name=='ours' else args.model_name
    with open(result_path,"w") as f:
        if model_checkpoint in result.keys():
            result[model_checkpoint].update({epoch_num:{'ImagePrec@1':df['ImagePrec@1'][0],"TextPrec@1":df['TextPrec@1'][0]}})
        else:
            result.update({model_checkpoint:{epoch_num:{'ImagePrec@1':df['ImagePrec@1'][0],"TextPrec@1":df['TextPrec@1'][0]}}})
        json.dump(result,f,sort_keys=True,indent=2)

    os.makedirs(output_file,exist_ok=True)
    df.to_csv(os.path.join(output_file,f"{args.dataset}.csv"))
    
if __name__ == "__main__":
    args = config()
    if args.model_name=='ours':
        assert args.resume is not None, "Test ours model must assign checkpont path"
    main(args)