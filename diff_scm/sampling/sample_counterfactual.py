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
import argparse
import random


from diff_scm.datasets import loader
from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util, script_util
from diff_scm.sampling.sampling_utils import get_models_functions, \
                            get_input_data, get_dict_of_arrays


def main(args):
    config = get_config.file_from_dataset(args.dataset)

    dist_util.setup_dist()
    logger.configure(Path(config.experiment_name) / ("counterfactual_sampling_" + "_".join(config.classifier.label)))

    logger.log("creating loader...")
    test_loader = loader.get_data_loader(args.dataset, config, split_set='test', generator = True) 

    logger.log("creating model and diffusion...")

    classifier, diffusion, model = script_util.get_models_from_config(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"Number of parameteres: {pytorch_total_params}")

    cond_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(config, model, classifier)

    logger.log("sampling...")

    all_results = []
    for i, data_dict in enumerate(test_loader):
        
        model_kwargs, init_image = get_input_data(config, data_dict)
        
        if config.sampling.reconstruction:
            latent_image, abduction_progression = diffusion.ddim_sample_loop(
                model_fn,
                (config.sampling.batch_size,
                    config.score_model.num_input_channels,
                    config.score_model.image_size,
                    config.score_model.image_size),
                clip_denoised=config.sampling.clip_denoised,
                model_kwargs=model_kwargs,
                denoised_fn = denoised_fn if config.sampling.dynamic_sampling else None,
                noise=init_image,
                cond_fn=None,
                device=dist_util.dev(),
                progress=config.sampling.progress,
                eta=config.sampling.eta,
                reconstruction=config.sampling.reconstruction,
                sampling_progression_ratio = config.sampling.sampling_progression_ratio
            )
            init_image = latent_image
        else:
            init_image = None

        counterfactual_image, diffusion_progression = diffusion.ddim_sample_loop(
            model_classifier_free_fn if config.score_model.classifier_free_cond else model_fn,
            (config.sampling.batch_size,
                config.score_model.num_input_channels,
                config.score_model.image_size,
                config.score_model.image_size),
            clip_denoised=config.sampling.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn = denoised_fn if config.sampling.dynamic_sampling else None,
            noise=init_image,
            cond_fn=cond_fn if config.sampling.classifier_scale != 0 else None,
            device=dist_util.dev(),
            progress=config.sampling.progress,
            eta=config.sampling.eta,
            reconstruction=False,
            sampling_progression_ratio = config.sampling.sampling_progression_ratio
        )
            
        results_per_sample = {"original": data_dict,
                            "counterfactual_sample" : counterfactual_image.cpu().numpy(),
                                                                }
        if False:
            results_per_sample = {"original": data_dict['image'].cpu().numpy(),
                                "gt": data_dict['gt'].cpu().numpy(),
                                "healthy_status": data_dict['y'].cpu().numpy(),
                                "patient_id" : data_dict['patient_id'].cpu().numpy(),
                                "slice_id" : data_dict['slice_id'].cpu().numpy(),
                                "counterfactual_sample" : counterfactual_image.cpu().numpy(),
                                                                }    
        if config.sampling.progress:
            results_per_sample.update({"diffusion_process": abduction_progression + diffusion_progression})
                                                        
        all_results.append(results_per_sample)

        if config.sampling.num_samples is not None:
            
            if ((i+1) * config.sampling.batch_size) >= config.sampling.num_samples:
                print( (1+i) * config.sampling.batch_size)
                break

    all_results = {k: [dic[k] for dic in all_results] for k in all_results[0]}

    #logger.log(f"created {all_results['original'].shape[0]} samples")

    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"samples_{config.sampling.label_of_intervention}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, all_results)

    dist.barrier()
    logger.log("sampling complete")

def reseed_random(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="mnist or brats", type=str)
    args = parser.parse_args()
    print(args.dataset)
    main(args)
