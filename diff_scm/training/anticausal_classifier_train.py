"""
Train a noised image classifier on ImageNet.
"""

import os
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch
import torchvision
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util
from diff_scm.utils.script_util import create_anti_causal_predictor, create_gaussian_diffusion
from diff_scm.utils.fp16_util import MixedPrecisionTrainer
from diff_scm.modelsOld.resample import create_named_schedule_sampler
from diff_scm.training.train_util import parse_resume_step_from_filename, log_loss_dict
from diff_scm.datasets import loader


def main(args):
    config = get_config.file_from_dataset(args.dataset)

    dist_util.setup_dist()
    logger.configure(Path(config.experiment_name) / ("classifier_train_" + "_".join(config.classifier.label)),
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(config)

    model = create_anti_causal_predictor(config)
    model.to(dist_util.dev())

    if config.classifier.training.noised:
        schedule_sampler = create_named_schedule_sampler(
            config.classifier.training.schedule_sampler, diffusion
        )

    logger.log("creating data loader...")

    data = loader.get_data_loader(args.dataset, config, split_set='train', generator = True) 
    val_data = loader.get_data_loader(args.dataset, config, split_set='val', generator = True)

    logger.log("training...")

    resume_step = 0
    if config.classifier.training.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(config.classifier.training.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {config.classifier.training.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    config.classifier.training.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=config.classifier.training.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=True,
    )

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=config.classifier.training.lr,
                weight_decay=config.classifier.training.weight_decay)
    if config.classifier.training.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(config.classifier.training.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        data_dict = next(data_loader)
        batch = data_dict["image"].to(dist_util.dev())
        if args.dataset == "brats":
            batch = torchvision.transforms.Resize(size=256)(batch)
        labels = data_dict["y"].to(dist_util.dev())

        # Noisy images
        if config.classifier.training.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        loss_dict = get_predictor_loss(model, labels, batch, t)
        loss = torch.stack(list(loss_dict.values())).sum()
        losses = {f"{prefix}_{loss_name}": loss_value.detach() for loss_name, loss_value in loss_dict.items()}
        # losses[f"{prefix}_acc@1"] = compute_top_k(logits, labels, k=1, reduction="none")
        log_loss_dict(diffusion, t, losses)

        del losses
        loss = loss.mean()
        if loss.requires_grad:
            mp_trainer.zero_grad()
            mp_trainer.backward(loss)

    for step in range(config.classifier.training.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * config.classifier.training.batch_size * dist.get_world_size(),
        )
        if config.classifier.training.anneal_lr:
            set_annealed_lr(opt, config.classifier.training.lr,
                            (step + resume_step) / config.classifier.training.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % config.classifier.training.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % config.classifier.training.log_interval:
            logger.dumpkvs()
        if (
                step
                and dist.get_rank() == 0
                and not (step + resume_step) % config.classifier.training.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def get_predictor_loss(model, labels, batch, t):
    output = model(batch, timesteps=t)
    loss_dict = {}
    '''if isinstance(output, Dict):
        assert len(output["latents"]) == 2
        scale = 0.05
        loss_dict["hsic"] = scale * hsic_normalized(output["latents"][0], output["latents"][1], False)
        for label_name, label_value in labels.items():
            loss_dict[label_name] = torch.nn.BCELoss()(torch.nn.Sigmoid()(output[label_name]), label_value, reduction="mean")
    else:
        # loss_dict["loss"] = torch.nn.BCELoss()(torch.nn.Sigmoid()(output), labels["gt"])
        # loss_dict["loss"] = F.cross_entropy(output, list(labels.values())[0], reduction="mean")'''
    loss_dict["loss"] = F.cross_entropy(output, labels)
    return loss_dict


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i: i + microbatch] if x is not None else None for x in args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="mnist or brats", type=str, default='mnist')
    args = parser.parse_args()
    print(args.dataset)
    main(args)
