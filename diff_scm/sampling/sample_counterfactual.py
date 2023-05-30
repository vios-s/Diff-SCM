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

from matplotlib import pyplot as plt

sys.path.append(str(Path.cwd()))
import argparse
import random


from diff_scm.datasets import loader
from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util, script_util
from diff_scm.sampling.sampling_utils import get_models_functions, estimate_counterfactual

def main(args):
    config = get_config.file_from_dataset(args.dataset)

    dist_util.setup_dist()
    logger.configure(Path(config.experiment_name) / ("counterfactual_sampling_" + "_".join(config.classifier.label)))

    logger.log("creating loader...")
    test_loader = loader.get_data_loader(args.dataset, config, split_set='test', generator = False) 

    logger.log("creating model and diffusion...")

    classifier, diffusion, model = script_util.get_models_from_config(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"Number of parameters: {pytorch_total_params}")

    cond_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(config, model, classifier)

    logger.log("sampling...")

    all_results = []
    for i, data_dict in enumerate(test_loader):
        
        counterfactual_image, sampling_progression = estimate_counterfactual(config, 
                                                diffusion, cond_fn, model_fn, 
                                                model_classifier_free_fn, denoised_fn, 
                                                data_dict)
            
        results_per_sample = {"original": data_dict,
                              "counterfactual_sample" : counterfactual_image.cpu().numpy(),
                                                                }

        if config.sampling.progress:
            results_per_sample.update({"diffusion_process": sampling_progression})
                                                        
        all_results.append(results_per_sample)

        if config.sampling.num_samples is not None and ((i+1) * config.sampling.batch_size) >= config.sampling.num_samples:
            break                

    all_results = {k: [dic[k] for dic in all_results] for k in all_results[0]}

    counter = all_results['counterfactual_sample']
    orig = all_results['original'][0]['image']
    x = 0

    o1 = orig[x][0][:][:]
    fig = plt.figure(figsize=(12., 12.))
    plt.imshow(o1, cmap='gray')
    plt.axis("off")
    plt.show()

    l1 = all_results['original'][0]['y']
    print(l1[x])

    c1 = counter[0][x][0][:][:]
    fig = plt.figure(figsize=(12., 12.))
    plt.imshow(c1, cmap='gray')
    plt.axis("off")
    plt.show()

    diff = abs(c1) - abs(o1.numpy())
    # print(diff)

    fig = plt.figure(figsize=(12., 12.))
    plt.imshow(diff, cmap='viridis')
    plt.axis("off")
    plt.show()

    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"samples.npz")
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
    parser.add_argument("--dataset", help="mnist or brats", type=str, default='mnist')
    args = parser.parse_args()
    print(args.dataset)
    main(args)
