"""
Like score_sampling.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

from datasets import loader
from configs import default_mnist_configs
from utils import logger, dist_util
from sampling.sampling_utils import get_models_from_config,get_models_functions


def main():
    config = default_mnist_configs.get_default_configs()

    dist_util.setup_dist()
    logger.configure(Path(config.experiment_name) / ("counterfactual_sampling_" + "_".join(config.classifier.label)))

    logger.log("creating loader...")
    test_loader = loader.get_data_loader(config.data.path, config.sampling.batch_size, split_set='test',
                                         which_label=config.classifier.label)
    if config.sampling.image_conditional:
        test_loader = loader.get_conditional_data_loader(test_loader, train=False)

    logger.log("creating model and diffusion...")

    classifier, diffusion, model = get_models_from_config(config)

    cond_fn, model_fn = get_models_functions(config, model, classifier)

    logger.log("sampling...")

    all_results = []
    while len(all_results) * config.sampling.batch_size < config.sampling.num_samples:
        data_dict = next(test_loader)
        results_per_sample = {"original": ((data_dict['image'] + 1) * 127.5).clamp(0, 255).to(
            th.uint8)}
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in data_dict.items()}
        init_image = data_dict['image'].to(dist_util.dev())

        logger.log(f"counterfactuals for class {config.sampling.counterfactual_class}")
        model_kwargs["y"] = (config.sampling.counterfactual_class * th.ones((config.sampling.batch_size,))).to(th.long).to(dist_util.dev())
        counterfactual = diffusion.ddim_sample_loop(
            model_fn,
            (config.sampling.batch_size,
                config.score_model.num_input_channels,
                config.score_model.image_size,
                config.score_model.image_size),
            clip_denoised=config.sampling.clip_denoised,
            model_kwargs=model_kwargs,
            noise=init_image,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            progress=True,
            reconstruction=config.sampling.reconstruction,
        )
        counterfactual_sample_for_class_rescale = ((counterfactual[-1]['sample'] + 1) * 127.5).clamp(0, 255).to(
            th.uint8)
        results_per_sample[
            f"counterfactual"] = counterfactual_sample_for_class_rescale.cpu().numpy()

        all_results.append(results_per_sample)

    logger.log(f"created {len(all_results) * config.sampling.batch_size} samples")

    out_path = os.path.join(logger.get_dir(), f"samples_{config.sampling.label_of_intervention}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, all_results)

    dist.barrier()
    logger.log("sampling complete")



if __name__ == "__main__":
    main()
