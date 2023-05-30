from typing import List

import ml_collections
import torch
from pathlib import Path
import os


def get_default_configs():
    config = ml_collections.ConfigDict()

    config.dataset_name = "MNIST"
    config.experiment_name = "exp_02_" + config.dataset_name
    
    use_gpus = "0"  # e.g. "0,1,2"
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpus
    # data
    config.data = data = ml_collections.ConfigDict()
    data.path = Path(r"../diff_scm/datasets/mnist") / config.dataset_name
    experiment_path = r"../experiment_data/"
    ## Diffusion parameters
    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.steps = 1000
    diffusion.learn_sigma = False
    diffusion.sigma_small = False
    diffusion.noise_schedule = "linear"
    # if use_kl AND rescale_learned_sigmas are False, it will use MSE
    diffusion.use_kl = False
    diffusion.rescale_learned_sigmas = False
    diffusion.predict_xstart = False
    diffusion.rescale_timesteps = False
    diffusion.timestep_respacing = "ddim100"  
    diffusion.conditioning_noise = "constant"  

    ## score model config
    config.score_model = score_model = ml_collections.ConfigDict()
    score_model.image_size = 28
    score_model.classifier_free_cond = False
    score_model.num_input_channels = 1
    score_model.num_channels = 32
    score_model.num_res_blocks = 1
    score_model.num_heads = 1
    score_model.num_heads_upsample = -1
    score_model.num_head_channels = -1
    score_model.learn_sigma = diffusion.learn_sigma
    score_model.attention_resolutions = ""  # 16
    
    attention_ds = []
    if score_model.attention_resolutions != "":
        for res in score_model.attention_resolutions.split(","):
            attention_ds.append(score_model.image_size // int(res))
    score_model.attention_ds = attention_ds

    score_model.channel_mult = (1, 2, 2)
    score_model.dropout = 0.1
    score_model.class_cond = False
    score_model.use_checkpoint = False
    score_model.use_scale_shift_norm = True
    score_model.resblock_updown = False
    score_model.use_fp16 = False
    score_model.use_new_attention_order = False
    score_model.num_classes = 10
    score_model.image_level_cond = False

    # score model training
    config.score_model.training = training_score = ml_collections.ConfigDict()
    training_score.iterations = 1#3e4
    training_score.schedule_sampler = "uniform" 
    training_score.lr = 1e-4
    training_score.weight_decay = 0.01
    training_score.lr_anneal_steps = 0
    training_score.batch_size = 256
    training_score.microbatch = -1  # -1 disables microbatches
    training_score.ema_rate = "0.9999"  # comma-separated list of EMA values
    training_score.log_interval = 50
    training_score.save_interval = 1000
    training_score.resume_checkpoint = ""
    training_score.use_fp16 = score_model.use_fp16
    training_score.fp16_scale_growth = 1e-3
    training_score.conditioning_variable = "y"
    training_score.cond_dropout_rate = 0.0

    ## classifier config

    config.classifier = classifier = ml_collections.ConfigDict()
    classifier.image_size = score_model.image_size
    classifier.label = ['class']
    assert isinstance(classifier.label, List)
    classifier.classifier_width = 32
    classifier.classifier_depth = 1
    classifier.in_channels = score_model.num_input_channels
    classifier.out_channels = [10]  # number of classes
    classifier.channel_mult = (1, 2, 4, 4)
    classifier.classifier_attention_resolutions = ""  # 16"32,16,8"
    classifier.classifier_use_scale_shift_norm = True  # False
    classifier.classifier_resblock_updown = True  # False
    classifier.classifier_pool =  "attention"
    classifier.classifier_use_fp16 = score_model.use_fp16
    classifier.noise_conditioning = True
    attention_ds = []
    if classifier.classifier_attention_resolutions != "":
        for res in classifier.classifier_attention_resolutions.split(","):
            attention_ds.append(classifier.image_size // int(res))
    classifier.attention_ds = tuple(attention_ds)

    config.classifier.training = training_class = ml_collections.ConfigDict()
    training_class.noised = True
    training_class.adversarial_training = False
    training_class.iterations = 3
    training_class.lr = 1e-4
    training_class.weight_decay = 0.0
    training_class.anneal_lr = False
    training_class.batch_size = 256
    training_class.microbatch = -1
    training_class.schedule_sampler = "uniform"
    training_class.resume_checkpoint = False  # ""
    training_class.log_interval = 100
    training_class.eval_interval = 50
    training_class.save_interval = 1000
    training_class.classifier_use_fp16 = score_model.use_fp16

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.clip_denoised = True
    sampling.dynamic_sampling = True
    sampling.progress = False
    sampling.num_samples = 1
    sampling.batch_size = 100
    sampling.use_ddim = True
    sampling.reconstruction = True
    sampling.eta = 0.0
    sampling.image_conditional = False
    sampling.label_of_intervention = "y" 
    sampling.model_path = experiment_path + config.experiment_name + "/score_train/model000020.pt"
    sampling.classifier_path = experiment_path + config.experiment_name + "/classifier_train_" + "_".join(config.classifier.label) + "/model000099.pt"
    sampling.classifier_scale = 1.0
    sampling.target_class = 5    
    sampling.sampling_progression_ratio = 0.75
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
