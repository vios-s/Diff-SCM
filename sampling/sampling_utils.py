from typing import Dict
import torch as th
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
from utils import logger, dist_util


from utils.script_util import create_gaussian_diffusion, create_image_cond_score_model, create_anti_causal_predictor, \
    create_score_model
from utils import logger, dist_util


def get_models_from_config(config):
    diffusion = create_gaussian_diffusion(config)
    if config.sampling.image_conditional:
        model = create_image_cond_score_model(config)
        model.load_state_dict(
            dist_util.load_state_dict(config.sampling.image_conditioned_model_path, map_location=dist_util.dev())
        )
    else:
        model = create_score_model(config)
        model.load_state_dict(
            dist_util.load_state_dict(config.sampling.model_path, map_location=dist_util.dev())
        )
    model.to(dist_util.dev())
    if config.score_model.use_fp16:
        model.convert_to_fp16()
    model.eval()
    classifier = create_anti_causal_predictor(config)
    classifier.load_state_dict(
        dist_util.load_state_dict(config.sampling.classifier_path, map_location=dist_util.dev())
    )
    classifier.to(dist_util.dev())
    if config.classifier.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    return classifier, diffusion, model


def get_models_functions(config, model, anti_causal_predictor):
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = anti_causal_predictor(x_in, t)
            if isinstance(out, Dict):
                logits = out[config.sampling.label_of_intervention]
            else:
                logits = out
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad_log_conditional = th.autograd.grad(selected.sum(), x_in)[0]
            return grad_log_conditional * config.sampling.classifier_scale 

    def model_fn(x, t, y=None, conditioning_x=None, **kwargs):
        return model(x, t, conditioning_x=conditioning_x)

    return cond_fn, model_fn
