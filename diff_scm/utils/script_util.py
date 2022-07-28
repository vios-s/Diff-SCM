import argparse
import ml_collections

from diff_scm.utils import dist_util
from diff_scm.models import gaussian_diffusion as gd
from diff_scm.models import unet
from diff_scm.models.respace import SpacedDiffusion, space_timesteps


def get_models_from_config(config):
    diffusion = create_gaussian_diffusion(config)
    model = create_score_model(config)
    model.load_state_dict(
        dist_util.load_state_dict(config.sampling.model_path, map_location=dist_util.dev())
    )
    model.to(dist_util.dev())
    if config.score_model.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    if config.sampling.classifier_scale != 0:
        classifier = create_anti_causal_predictor(config)
        classifier.load_state_dict(
            dist_util.load_state_dict(config.sampling.classifier_path, map_location=dist_util.dev())
        )
        classifier.to(dist_util.dev())
        if config.classifier.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
    else:
        classifier = None
    return classifier, diffusion, model


def create_score_model(config: ml_collections.ConfigDict):
    return unet.UNetModel(
        in_channels=config.score_model.num_input_channels,
        model_channels=config.score_model.num_channels,
        out_channels=(
            config.score_model.num_input_channels
            if not config.score_model.learn_sigma else 2 * config.score_model.num_input_channels),
        num_res_blocks=config.score_model.num_res_blocks,
        attention_resolutions=tuple(config.score_model.attention_ds),
        dropout=config.score_model.dropout,
        channel_mult=config.score_model.channel_mult,
        num_classes=(config.score_model.num_classes if config.score_model.class_cond else None),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=config.score_model.num_heads,
        num_head_channels=config.score_model.num_head_channels,
        num_heads_upsample=config.score_model.num_heads_upsample,
        use_scale_shift_norm=config.score_model.use_scale_shift_norm,
        resblock_updown=config.score_model.resblock_updown,
        image_level_cond=config.score_model.image_level_cond,
    )

def create_anti_causal_predictor(config):
    enc = []
    nb_variables = len(config.classifier.label)
    for i in range(nb_variables):
        enc.append(unet.EncoderUNetModel(
            image_size=config.classifier.image_size,
            in_channels=config.classifier.in_channels,
            model_channels=config.classifier.classifier_width,
            out_channels=config.classifier.out_channels[i] if nb_variables == 1 else 128,
            num_res_blocks=config.classifier.classifier_depth,
            attention_resolutions=config.classifier.attention_ds,
            channel_mult=config.classifier.channel_mult,
            use_fp16=config.classifier.classifier_use_fp16,
            num_head_channels=64,
            use_scale_shift_norm=config.classifier.classifier_use_scale_shift_norm,
            resblock_updown=config.classifier.classifier_resblock_updown,
            pool=config.classifier.classifier_pool,
        ))
    if nb_variables == 1:
        model = enc[0]
    else:
        model = unet.AntiCausalMechanism(encoders=enc, out_labels=config.classifier.label,
                                    out_channels=config.classifier.out_channels)
    print(model)
    return model



def create_gaussian_diffusion(config):
    betas = gd.get_named_beta_schedule(config.diffusion.noise_schedule, config.diffusion.steps)
    if config.diffusion.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif config.diffusion.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    timestep_respacing = config.diffusion.timestep_respacing
    if not timestep_respacing:
        timestep_respacing = [config.diffusion.steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(config.diffusion.steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not config.diffusion.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not config.diffusion.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not config.diffusion.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=config.diffusion.rescale_timesteps,
        conditioning_noise=config.diffusion.conditioning_noise
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
