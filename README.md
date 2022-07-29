# Diff-SCM

## Overview

Diff-SCM aims at developing solutions based diffusion probabilistic modeling (DPM) for causal machine learning. Our [algorithm](https://arxiv.org/abs/2202.10166) perform counterfactual inference using Pearl's abduction-action-prediction strategy using a deterministic inference based on a deterministic diffusion inference ([DDIM](https://arxiv.org/abs/2010.02502)). We have also [applied](https://arxiv.org/abs/2207.12268) to the method for lesion localization in Brain MRI.

The code is this repo implements the following papers:

> Sanchez, Pedro, and Sotirios A. Tsaftaris. "Diffusion Causal Models for Counterfactual Estimation." In *Conference on Causal Learning and Reasoning* (CLeaR). 2022.

> Sanchez, Pedro, Kascenas, Antanas, Liu, Xiao, O'Neil, Alison  and Sotirios A. Tsaftaris. "What is Healthy? Generative Counterfactual Diffusion for Lesion Localization." In *Deep Generative Models workshop at MICCAI 2022* (DGM4Miccai). 2022.

Note. If one is looking for particular bits of code with our contributions, consider checking the `diff_scm/sampling/sampling_utils.py` file. The code for counterfactual estimation, classfier-free (implicit) inference and dynamic normalisation is there.

## How to run

### Install

The provided `.yml` file can be use for creating a conda environment with the following command.

`conda env create -f environment.yml`

### Hyperparameters

A few hyperparameters are particularly important for counterfactual estimation. In particular, the parameters the control the strenght of the intervention. The `sampling.classifier_scale`, `sampling.norm_cond_scale`, `sampling.sampling_progression_ratio` are the some of these and can be found in the `diff_scm/configs` folder. Also, se this folder for example configuration on MNIST and BRATS dataset. 

