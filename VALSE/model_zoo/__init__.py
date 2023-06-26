import os
import clip
from PIL import Image
from torchvision import transforms
from .constants import CACHE_DIR
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor

def get_model(model_name, device,resume=None, root_dir=CACHE_DIR):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        print('testing openai clip model')
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "blip-flickr-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-flickr-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-coco-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-coco-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "xvlm-flickr":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-flickr")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-coco":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-coco", pretrained=resume)
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    elif model_name == "xvlm-16m":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-16m")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "flava":
        from .flava import FlavaWrapper
        flava_model = FlavaWrapper(root_dir=root_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif model_name == "NegCLIP":
        import open_clip
        from .clip_models import CLIPWrapper
        
        path = os.path.join(root_dir, "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    elif model_name=="cyclip":
        import open_clip
        from .clip_models import CLIPWrapper
        print(f"test cyclip model, with checkpoints at {resume}")
        path=resume
        model, _, image_preprocess = open_clip.create_model_and_transforms('RN50', pretrained=path, device=device)
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess
    elif model_name == "ours":
        print(f'testing ours model, with checkpoints at {resume}')
        import open_clip
        from .clip_models import CLIPWrapper
        path=resume
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    elif model_name == "coca":
        import open_clip
        from .clip_models import CLIPWrapper
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name="coca_ViT-B-32", pretrained="laion2B-s13B-b90k", device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    elif "laion-clip" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name=variant, pretrained="laion2b_s34b_b79k", device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    else:
        raise ValueError(f"Unknown model {model_name}")
