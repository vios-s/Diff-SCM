"""
Train a diffusion model on images.
"""
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import argparse
from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util
from diff_scm.modelsOld.resample import create_named_schedule_sampler
from diff_scm.utils.script_util import create_gaussian_diffusion, create_score_model
from diff_scm.training.train_util import TrainLoop
from diff_scm.datasets import loader


def main(args):
    config = get_config.file_from_dataset(args.dataset)

    dist_util.setup_dist()
    logger.configure(Path(config.experiment_name)/"score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(config)
    model = create_score_model(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(dist_util.dev())
    logger.log(f"Model number of parameters {pytorch_total_params}")
    schedule_sampler = create_named_schedule_sampler(config.score_model.training.schedule_sampler, diffusion)


    logger.log("creating data loader...")
    train_loader = loader.get_data_loader(args.dataset, config, split_set='train', generator = True) 
    val_loader = loader.get_data_loader(args.dataset, config, split_set='val', generator = True)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader ,
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
        cond_dropout_rate = config.score_model.training.cond_dropout_rate if config.score_model.class_cond else 0,
        conditioning_variable = config.score_model.training.conditioning_variable,
        iterations = config.score_model.training.iterations
    ).run_loop()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="mnist or brats", type=str,default='mnist')
    args = parser.parse_args()
    print(args.dataset)
    main(args)
