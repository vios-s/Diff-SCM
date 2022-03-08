"""
Train a diffusion model on images.
"""
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

from configs import default_mnist_configs
from utils import logger, dist_util
from models.resample import create_named_schedule_sampler
from utils.script_util import create_gaussian_diffusion,create_score_model
from training.train_util import TrainLoop
from datasets import loader


def main():
    config = default_mnist_configs.get_default_configs()

    dist_util.setup_dist()
    logger.configure(Path(config.experiment_name)/"score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(config)
    model = create_score_model(config)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(config.score_model.training.schedule_sampler, diffusion)


    logger.log("creating data loader...")
    train_loader = loader.get_data_loader(config.data.path, config.score_model.training.batch_size, split_set='train')
    val_loader = loader.get_data_loader(config.data.path, config.score_model.training.batch_size, split_set='val')

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        data_val=val_loader,
        batch_size=config.score_model.training.batch_size,
        microbatch=config.score_model.training.microbatch,
        lr=config.score_model.training.lr,
        ema_rate=config.score_model.training.ema_rate,
        log_interval=config.score_model.training.log_interval,
        save_interval=config.score_model.training.save_interval,
        resume_checkpoint=config.score_model.training.resume_checkpoint,
        use_fp16=config.score_model.training.use_fp16,
        fp16_scale_growth=config.score_model.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=config.score_model.training.weight_decay,
        lr_anneal_steps=config.score_model.training.lr_anneal_steps,
    ).run_loop(config.score_model.training.iterations)


if __name__ == "__main__":
    main()
