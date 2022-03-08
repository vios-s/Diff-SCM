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

from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

from configs import default_mnist_configs
from utils import logger, dist_util
from utils.script_util import create_anti_causal_predictor, create_gaussian_diffusion
from utils.fp16_util import MixedPrecisionTrainer
from models.resample import create_named_schedule_sampler
from training.train_util import parse_resume_step_from_filename, log_loss_dict
from datasets import loader


def main():
    config = default_mnist_configs.get_default_configs()

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
    data = loader.get_data_loader(config.data.path, config.classifier.training.batch_size, split_set='train',
                                  which_label=config.classifier.label)
    val_data = loader.get_data_loader(config.data.path, config.classifier.training.batch_size, split_set='val',
                                      which_label=config.classifier.label)
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
        find_unused_parameters=False,
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
        labels = {}
        for label_name in config.classifier.label:
            assert label_name in list(data_dict.keys()), f'label {label_name} are not in data_dict{data_dict.keys()}'
            labels[label_name] = data_dict[label_name].to(dist_util.dev())

        batch = data_dict["image"].to(dist_util.dev())

        # Noisy images
        if config.classifier.training.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        loss_dict = get_predictor_loss(model, labels, batch, t)
        loss = torch.stack(list(loss_dict.values())).sum()
        losses = {f"{prefix}_{loss_name}": loss_value.detach() for loss_name, loss_value in loss_dict.items()}
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
    loss_dict["loss"] = F.cross_entropy(output, list(labels.values())[0], reduction="mean")
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


"""
         for i, (sub_batch, sub_labels, sub_t) in enumerate(
     split_microbatches(config.classifier.training.microbatch, batch, labels, t)
):
 if not config.classifier.noise_conditioning:
     sub_t = None

 if prefix == "train" and config.classifier.training.adversarial_training:
     sub_batch_perturbed = adversarial_attacker.perturb(model, sub_batch, sub_labels, sub_t)
     logits_perturbed = model(sub_batch_perturbed, timesteps=sub_t)
     loss += F.cross_entropy(logits_perturbed, sub_labels, reduction="none")
     loss /= 2
     adversarial_sub_labels = get_random_vector_excluding(sub_labels)
 adversarial_sub_batch = fgsm_attack(sub_batch, sub_batch.grad.data)
 adversarial_logits = model(adversarial_sub_batch, timesteps=sub_t)
 """


# FGSM attack code
def fgsm_attack(original_batch, data_grad, epsilon: float = 0.15):
    epsilon = th.tensor(epsilon).to(data_grad.device)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_batch = original_batch + epsilon * sign_data_grad
    # Adding clipping to maintain [-1,1] range
    perturbed_batch = th.clamp(perturbed_batch, -1, 1)
    # Return the perturbed image
    return perturbed_batch


if __name__ == "__main__":
    main()
