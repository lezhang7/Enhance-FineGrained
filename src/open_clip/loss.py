import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None



def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss



def gather_features_da(
        image_features,
        text_features,
        valid_caption_mask,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_valid_caption_mask=torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                gathered_valid_caption_mask[rank] = valid_caption_mask
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)

    return all_image_features, all_text_features, all_valid_caption_mask

class Clip_DALoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            rank_loss=False,
            discriminative_loss=False,
            hardnegative=False,
            dis_loss_weight=0.2,
            rank_loss_weight=0.2,
            threshold='mean'
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.rank_loss=rank_loss
        self.discriminative_loss=discriminative_loss
        self.dis_loss_weigth=dis_loss_weight
        self.rank_loss_weight=rank_loss_weight
        self.threshold=threshold
        self.hardnegative=hardnegative
        
    def forward(self, image_features, text_features,valid_caption_mask, logit_scale, thresholds):
        """
        ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hardnegative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - discriminative_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - rank_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        device = image_features.device
        r_loss,d_loss=0.0,0.0
        thresholds_new=None
        thresholds=2 if thresholds is None else thresholds
        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size
            da_all_text_features=all_text_features[caption_types==2] # 4 * batch_size * word_size
            gt_len,feature_size=all_image_features.shape[0],all_image_features.shape[-1]


            # print('-'*50)
            # print(f"all_image_features.shape :{all_image_features.shape}")
            # print(f"all_text_features.shape :{all_text_features.shape}")
            # print(f"gt_len.shape :{gt_len}, feature_size: {feature_size}")
            # print(f"all_valid_caption_mask.shape :{all_valid_caption_mask.shape}")

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                #extra hard negative loss
                if self.hardnegative:
                    all_text_features=torch.cat([gt_all_text_features,da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T # batch_size * 4xbatch_size
                    # applying this would make inferior performance
                    # hardnegative_caption_mask=torch.cat((torch.ones(gt_all_text_features.shape[0],device=all_valid_caption_mask.device),all_valid_caption_mask.flatten()))
                    # hardnegative_caption_mask=((hardnegative_caption_mask-1)*99999).unsqueeze(0).expand(logits_per_image.shape)
                    # logits_per_image+=hardnegative_caption_mask
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T

                # rank loss
                if self.rank_loss:
                    da_logits_per_image= logit_scale * (da_all_text_features.reshape(gt_len,-1,feature_size)@ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    r_loss,thresholds_new=self.rank_clip_loss(logits_per_image,da_logits_per_image,all_valid_caption_mask,thresholds)
                
                # discriminative loss
                if self.discriminative_loss in ["text","all"]:
                    text_embedding_matrix=logit_scale * gt_all_text_features @ da_all_text_features.T  #(all_batch_size,4*all_batch_size)
                    # dis_caption_mask=((all_valid_caption_mask.flatten()-1)*99999).unsqueeze(0).expand(text_embedding_matrix.shape)
                    # text_embedding_matrix+=dis_caption_mask
                    d_loss+=self.discriminative_clip_loss(logits_per_image,text_embedding_matrix)
                if self.discriminative_loss in ["image","all"]:
                    image_embedding_matrix=logit_scale* all_image_features@all_image_features.T
                    n,m=image_embedding_matrix.shape
                    assert n==m
                    image_embedding_matrix_off_diag=image_embedding_matrix.flatten()[:-1].view(n-1,n+1)[:,1:].flatten().reshape(image_embedding_matrix.shape[0],-1)
                    d_loss+=self.discriminative_clip_loss(logits_per_image,image_embedding_matrix_off_diag)

        else:
        # not updating very long time
            gt_len,feature_size=image_features.shape[0],image_features.shape[-1]
            gt_text_features=text_features[:image_features.shape[0]]
            da_text_features=text_features[image_features.shape[0]:]
            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ gt_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * gt_text_features @ image_features.T
            if self.rank_loss:
                da_logits_per_image=  logit_scale * (da_text_features.reshape(gt_len,-1,feature_size)@ image_features.unsqueeze(-1)).squeeze() * valid_caption_mask
                r_loss,thresholds_new=self.rank_clip_loss(logits_per_image,da_logits_per_image,valid_caption_mask,thresholds)
            if self.discriminative_loss in ["text","all"]:
                #contrastive loss between caption and hard-negative captions
                text_embedding_matrix=logit_scale * gt_text_features @ da_text_features.T #(batch_size,4*batch_size)
                d_loss=self.discriminative_clip_loss(logits_per_image,text_embedding_matrix)
            if self.discriminative_loss in ["image","all"]:
                #contrastive loss between images
                image_embedding_matrix=logit_scale* image_features@image_features.T
                n,m=image_embedding_matrix.shape
                assert n==m
                image_embedding_matrix_off_diag=image_embedding_matrix.flatten()[:-1].view(n-1,n+1)[:,1:].flatten().reshape(image_embedding_matrix.shape[0],-1)
                d_loss+=self.discriminative_clip_loss(logits_per_image,image_embedding_matrix_off_diag)
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        if self.rank_loss:
            total_loss+=r_loss*self.rank_loss_weight
        if self.discriminative_loss:
            total_loss+=d_loss*self.dis_loss_weigth
            
        return total_loss,thresholds_new,r_loss,d_loss
     
        
    def rank_clip_loss(self,gt_logits_per_image:torch.Tensor,da_logits_per_image:torch.Tensor,valid_caption_mask,thresholds:torch.Tensor) -> torch.Tensor:
        gt_similarity=gt_logits_per_image.gather(0,torch.arange(min(gt_logits_per_image.shape),device=gt_logits_per_image.device).reshape(1,-1)).reshape(min(gt_logits_per_image.shape),1).expand(da_logits_per_image.shape)
        rank_loss=nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))*valid_caption_mask

        if self.threshold=='mean':
            mask = da_logits_per_image!=0
            average_similarity_for_types = (da_logits_per_image*mask).sum(dim=0)/mask.sum(dim=0)
            thresholds_new=(gt_similarity.mean(0)-average_similarity_for_types).expand(gt_similarity.shape)
        elif self.threshold=='max':
            thresholds_new,max_indices=(gt_similarity*valid_caption_mask-da_logits_per_image).max(0)
            thresholds_new=thresholds_new.expand(gt_similarity.shape)/5
        
        return rank_loss.mean(),thresholds_new.detach()

    def discriminative_clip_loss(self,gt_logits_per_image:torch.Tensor,embedding_matrix:torch.Tensor):
        """
        gt_logits_per_image: standard clip similarity matrix, diag is true gt similarity value : shape [batch_size,batch_size]
        embedding_matrix: extra similarity matrix served as denominator in clip loss
        """
        gt_similarity=gt_logits_per_image.diag().reshape(-1,1)
        logtis_matrix=torch.cat([gt_similarity,embedding_matrix],dim=-1)
        labels=torch.ones(logtis_matrix.shape[0],device=gt_similarity.device,dtype=torch.long)
        d_loss=F.cross_entropy(logtis_matrix,labels)
        return d_loss
        