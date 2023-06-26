import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import Clip_DALoss, get_cast_dtype,ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.extra_da:
        loss = Clip_DALoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            atr_loss=args.atr_loss,
            tec_loss=args.tec_loss,
            tec_loss_weight=args.tec_loss_weight,
            atr_loss_weight=args.atr_loss_weight,
            threshold_type=args.threshold_type,
            hardnegative=args.hardnegative,
           
            )
    else:
        loss=ClipLoss(local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod
            )



    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    if args.atr_loss and args.threshold_type=="fixed":
        # init thresholds for fixed threshold
        thresholds=args.fixed_threshold_value
    elif args.atr_loss:
        # init thresholds for mean atr loss
        thresholds=0.0
    else:
        # without atr loss
        thresholds=None 

    atr_loss=None
    tec_loss=None
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        if args.extra_da:
            images, texts, valid_caption_mask = batch
            valid_caption_mask=valid_caption_mask[:,1:]
            valid_caption_mask=valid_caption_mask.to(device=device, non_blocking=True)
        else:
            images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                image_features, text_features, logit_scale = model(images, texts)
                if args.extra_da:
                    total_loss, thresholds ,atr_loss,tec_loss = loss(image_features, text_features,valid_caption_mask, logit_scale,thresholds)
                    if args.threshold_type!="fixed" and thresholds is not None and args.upper_bound is not None:
                        thresholds=torch.clamp(thresholds,0,args.upper_bound)
                else:
                    total_loss=loss(image_features, text_features, logit_scale)
            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this malocalkes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                    text_features = torch.cat(
                        accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
                    total_loss = loss(image_features, text_features, logit_scale)
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"ATR_Loss: {atr_loss if atr_loss is not None else None} "
                f"TEC_Loss: {tec_loss if tec_loss is not None else None} "
            )
              
            if args.atr_loss and args.threshold_type=="fixed":
                # init thresholds for fixed threshold
                logging.info(f"Thresholds: {thresholds}")
            elif args.atr_loss:
                # init thresholds for mean atr loss
                logging.info(f"Threshold_1: {thresholds.mean(0)[0].item() }"
                             f"Threshold_2: {thresholds.mean(0)[1].item() }"
                             f"Threshold_3: {thresholds.mean(0)[2].item() }"
                             f"Threshold_4: {thresholds.mean(0)[3].item() }")
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "ATR_Loss": atr_loss if atr_loss is not None else None,
                "TEC_Loss": tec_loss if tec_loss is not None else None
            }
             
            if args.atr_loss and args.threshold_type=="fixed":
                # init thresholds for fixed threshold
                log_data['threshold']=thresholds
            elif args.atr_loss:
                # init thresholds for mean atr loss
                log_data["threshold_1"]= thresholds.mean(0)[0].item() 
                log_data["threshold_2"]= thresholds.mean(0)[1].item() 
                log_data["threshold_3"]= thresholds.mean(0)[2].item()
                log_data["threshold_4"]= thresholds.mean(0)[3].item()

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def calculate_winoground_score(winoground_clip_scores):
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0
        
    denominator = len(winoground_clip_scores)
    # print("text score:", text_correct_count/denominator)
    # print("image score:", image_correct_count/denominator)
    # print("group score:", group_correct_count/denominator)
    # wandb.log({'text score':text_correct_count/denominator,"image score":image_correct_count/denominator,"group score":group_correct_count/denominator})
    return {"text score":text_correct_count/denominator,"image score":image_correct_count/denominator,"group score":group_correct_count/denominator}
def evaluate_winoground(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        scores={'c0_i0':[],'c0_i1':[],'c1_i0':[],'c1_i1':[]}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                batch_size=texts.shape[0]//2
                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    image_features=image_features.view(batch_size,2,-1)
                    text_features=text_features.view(batch_size,2,-1)
                    logits_per_image = torch.matmul(image_features,text_features.transpose(1,2)) * logit_scale

                    logits_per_image=logits_per_image.squeeze().cpu()
                    clip_score_c0_i0=logits_per_image[:,0,0].tolist()
                    clip_score_c1_i0=logits_per_image[:,0,1].tolist()
                    clip_score_c0_i1=logits_per_image[:,1,0].tolist()
                    clip_score_c1_i1=logits_per_image[:,1,1].tolist()
                    scores['c0_i0'].extend(clip_score_c0_i0)
                    scores['c0_i1'].extend(clip_score_c0_i1)
                    scores['c1_i0'].extend(clip_score_c1_i0)
                    scores['c1_i1'].extend(clip_score_c1_i1)

                    # logits_per_text = logits_per_image.t()

                    # batch_size = images.shape[0]
                    # labels = torch.arange(batch_size, device=device).long()
                    # total_loss = (
                    #     F.cross_entropy(logits_per_image, labels) +
                    #     F.cross_entropy(logits_per_text, labels)
                    # ) / 2

                # cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                # if is_master(args) and (i % 100) == 0:
                #     logging.info(
                #         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                #         f"Loss: {cumulative_loss / num_samples:.6f}\t")
            winoground_clip_scores=[]
            for idx in range(len(scores['c0_i0'])):
                winoground_clip_scores.append({"id" : idx, "c0_i0": scores['c0_i0'][idx], "c0_i1": scores['c0_i1'][idx], "c1_i0": scores['c1_i0'][idx], "c1_i1": scores['c1_i1'][idx]})
            winoground_metric=calculate_winoground_score(winoground_clip_scores)
            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            # loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics,**winoground_metric, "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
