import numpy as np
from torch import nn
import torch
import yaml
import os



def compute_similarity(image_features, text_features, split=100, bs = 1000):
    # compute similarity
    image_len = image_features.shape[0]
    text_len  = text_features.shape[0]
    total_ranks=np.zeros(image_len)
    for eid,v in enumerate(range(0, image_len, split)):
        partial_image_features = image_features[v:v+split]
        max_pairs=partial_image_features.shape[0]
        sub_similarity_scores = np.zeros((max_pairs, text_len))
        for v in range(0, max_pairs, bs):
            for t in range(0, text_len, bs):
                batch_visual_emb = image_features[v:v+bs]
                batch_caption_emb = text_features[t:t+bs]
                logits = batch_visual_emb @ batch_caption_emb.T
                sub_similarity_scores[v:v+bs,t:t+bs] = logits
        rank,top1 = compute_retrieval(sub_similarity_scores.numpy(),eid*split)
        total_ranks=np.concatenate((total_ranks,rank))
    r1 = 100.0 * len(np.where(total_ranks < 1)[0]) / len(total_ranks)
    r5 = 100.0 * len(np.where(total_ranks < 5)[0]) / len(total_ranks)
    r10 = 100.0 * len(np.where(total_ranks < 10)[0]) / len(total_ranks)
    r50 = 100.0 * len(np.where(total_ranks < 50)[0]) / len(total_ranks)
    medr = np.floor(np.median(total_ranks)) + 1
    meanr = total_ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}
    return report_dict

def compute_retrieval(a2b_sims, return_ranks=True ,start_index=0):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(start_index,npts+start_index):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]
    return ranks, top1



# read from file
features=np.load("/home/mila/l/le.zhang/scratch/open_clip/src/itr_results/saved_features/rank_coco-dis_text_mean-hn--5e-06-weightd0.5-weightr0.2-ub5-w_special_epoch_1_features.npz")
text_features=features["caption_features"]
image_features=features["image_features"]
single_caption=True
if not single_caption:
    for cap_idx in range(text_features.shape[1]):
        similarity_scores = compute_similarity(image_features, text_features[:,cap_idx,:])
        i2t_dict = compute_retrieval(similarity_scores.numpy())
        print(cap_idx, 'i2t', i2t_dict)
else:
    similarity_scores = compute_similarity(image_features, text_features)
    i2t_dict = compute_retrieval(similarity_scores.numpy())
    print('i2t', i2t_dict)