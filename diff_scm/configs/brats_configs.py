from typing import List

import ml_collections
import torch
from pathlib import Path
import os


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.dataset_name = "BRATS" 
    dataset = "brats2021_64x64" # "brats2021_64x64" , "brats2021_preprocessed_99max"
    use_gpus = "0"  # e.g. "0,1,2"
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpus
    # data
    config.data = data = ml_collections.ConfigDict()
    data.path = Path("/remote/rds/groups/idcom_imaging/data/Brain/BRATS/") / dataset
    data.sequence_translation = False # bool
    data.healthy_data_percentage = 1.0  # float [0,1]; 1 for training using full data; None for training with healthy data only
    #config.experiment_name = f"anomaly_diffusion_healthy_{data.healthy_data_percentage}_oversample_" + dataset
    config.experiment_name = f"anomaly_diffusion_classfree_attention_run1_" + dataset
    #config.experiment_name = f"anomaly_diffusion_CAM_" + dataset
    #config.experiment_name = "anomaly_diffusion_healthy_only_train_" + dataset
    #config.experiment_name = f"anomaly_diffusion_healthy_only_train_" + dataset
    experiment_path = r"/home/s2086085/RDS/experiment_data/"

    ## Diffusion parameters
    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.steps = 1000
    diffusion.learn_sigma = False
    diffusion.sigma_small = False
    diffusion.noise_schedule = "linear" #linear, cosine
    # if use_kl AND rescale_learned_sigmas are False, it will use MSE
    diffusion.use_kl = False
    diffusion.rescale_learned_sigmas = False
    diffusion.predict_xstart = False
    diffusion.rescale_timesteps = False
    diffusion.timestep_respacing = "ddim100"  # "" or "ddim25"
    diffusion.conditioning_noise = "constant"  # "constant" or "reverse"

    ## score model config
    config.score_model = score_model = ml_collections.ConfigDict()
    score_model.image_size = 64
    score_model.num_input_channels = 4
    score_model.num_channels = 64 #64,96
    score_model.num_res_blocks = 2 #2,3
    score_model.num_heads = 1
    score_model.num_heads_upsample = -1
    score_model.num_head_channels = -1
    score_model.learn_sigma = diffusion.learn_sigma
    score_model.attention_resolutions = "32,16,8"  # 16

    attention_ds = []
    if score_model.attention_resolutions != "":
        for res in score_model.attention_resolutions.split(","):
            attention_ds.append(score_model.image_size // int(res))
    score_model.attention_ds = attention_ds

    score_model.channel_mult = {64:(1, 2, 3, 4), 128:(1, 1, 2, 3, 4)}[score_model.image_size]
    score_model.dropout = 0.1
    score_model.use_checkpoint = False
    score_model.use_scale_shift_norm = True
    score_model.resblock_updown = True
    score_model.use_fp16 = False
    score_model.use_new_attention_order = False

    # score model training
    config.score_model.training = training_score = ml_collections.ConfigDict()
    training_score.schedule_sampler = "uniform"  # "uniform" or "loss-second-moment"
    training_score.lr = 1e-4
    training_score.weight_decay = 0.01
    training_score.lr_anneal_steps = 0
    training_score.batch_size = 64
    training_score.microbatch = -1  # -1 disables microbatches
    training_score.ema_rate = "0.9999"  # comma-separated list of EMA values
    training_score.log_interval = 50
    training_score.save_interval = 10000
    training_score.resume_checkpoint = ""
    training_score.use_fp16 = score_model.use_fp16
    training_score.fp16_scale_growth = 1e-3
    training_score.conditioning_variable = "gt"
    training_score.iterations = 50e3

    ## classifier config

    config.classifier = classifier = ml_collections.ConfigDict()
    classifier.image_size = score_model.image_size
    classifier.label = ['gt']
    assert isinstance(classifier.label, List)
    classifier.classifier_width = 64
    classifier.classifier_depth = 2
    classifier.in_channels = score_model.num_input_channels
    classifier.out_channels = [2]  # number of classes
    classifier.channel_mult = (1, 2, 4, 4)
    classifier.classifier_attention_resolutions = "" # 16"32,16,8"
    classifier.classifier_use_scale_shift_norm = True  # False
    classifier.classifier_resblock_updown = True  # False
    classifier.classifier_pool =  "attention" # adaptive
    classifier.classifier_use_fp16 = score_model.use_fp16
    classifier.noise_conditioning = False
    attention_ds = []
    if classifier.classifier_attention_resolutions != "":
        for res in classifier.classifier_attention_resolutions.split(","):
            attention_ds.append(classifier.image_size // int(res))
    classifier.attention_ds = tuple(attention_ds)

    config.classifier.training = training_class = ml_collections.ConfigDict()
    training_class.noised = False
    training_class.adversarial_training = False
    training_class.iterations = 20000
    training_class.lr = 1e-4
    training_class.weight_decay = 0.0
    training_class.anneal_lr = False
    training_class.batch_size = 64
    training_class.microbatch = -1
    training_class.schedule_sampler = "uniform"
    training_class.resume_checkpoint = False  # ""
    training_class.log_interval = 100
    training_class.eval_interval = 50
    training_class.save_interval = 1000
    training_class.classifier_use_fp16 = score_model.use_fp16

    # score conditioning
    score_model.class_cond = True
    score_model.num_classes = 2
    score_model.classifier_free_cond = True
    score_model.image_level_cond = False
    training_score.cond_dropout_rate = 0.35

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.clip_denoised = True
    sampling.dynamic_sampling = True
    sampling.num_samples = 32 # 1024, 34986
    sampling.batch_size = 32
    sampling.use_ddim = True
    sampling.reconstruction = True
    sampling.eta = 0.0
    sampling.label_of_intervention = "gt"
    sampling.conditioning_str = f"icond{config.score_model.image_level_cond}_ccond{config.score_model.class_cond}"
    sampling.model_path =  experiment_path + config.experiment_name + f"/score_train_{sampling.conditioning_str}/model050000.pt"
    sampling.classifier_path = experiment_path + config.experiment_name + "/classifier_train_" + "_".join(
        config.classifier.label) + "/model019000.pt"
    sampling.classifier_scale = 0.0
    sampling.norm_cond_scale = 3.0
    sampling.sampling_progression_ratio = 0.75
    sampling.sdedit_inpaint = False
    sampling.detection = True
    sampling.source_class = 1 # int in range [0, num_class-1]
    sampling.target_class = 0 # int in range [0, num_class-1]
    sampling.sampling_str = f"sampling_{sampling.conditioning_str}_nsamples{sampling.num_samples}_cfs{sampling.norm_cond_scale}_cs{sampling.classifier_scale}_spr{sampling.sampling_progression_ratio}"
    sampling.progress = True
    config.seed = 1
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

