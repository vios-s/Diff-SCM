from typing import Dict
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import numpy as np
from diff_scm.utils import logger, dist_util


def get_models_functions(config, model, anti_causal_predictor):
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = anti_causal_predictor(x_in, t)
            if isinstance(out, Dict):
                logits = out[config.sampling.label_of_intervention]
            else:
                logits = out
            #log_probs = torch.nn.Sigmoid()(logits)
            #selected = torch.nn.BCELoss(reduction='none')(log_probs, y.unsqueeze(1).to(th.float))#
            
            ## deal with
            y_new = torch.cat(2*[y[:y.size()[0]//2]]) if y.max() >= logits.size()[-1] else y
                
            log_probs = F.log_softmax(logits, dim=-1)
            selected =  log_probs[range(len(logits)), y_new.view(-1)]
            grad_log_conditional = torch.autograd.grad(selected.sum(), x_in)[0]

            # grad_channel_mean = grad_log_conditional.mean(tuple(range(1, len(grad_log_conditional.size()))))
            # scaling = (-selected / torch.exp(selected))
            # print({"selected": selected, "scaling": scaling, "grad_channel_mean": grad_channel_mean})
            return grad_log_conditional * config.sampling.classifier_scale  # * scaling[:, None, None, None]

    def model_fn(x, t, y=None, conditioning_x=None, **kwargs):
        # assert conditioning_x is not None
        y = (config.score_model.num_classes * torch.ones((config.sampling.batch_size,))).to(torch.long).to(dist_util.dev())
        return model(x, t, y = y, conditioning_x=conditioning_x)
        # Create an classifier-free guidance sampling function
    
    def model_classifier_free_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + config.sampling.norm_cond_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps

    def model_classifier_free_opt_fn(x_t, ts, **kwargs):
        ## conditional diffusion output
        cond_eps = model(x_t, ts, **kwargs)
        ## unconditional diffusion output
        uncond_kwargs = kwargs.copy()
        uncond_kwargs["y"] = config.score_model.num_classes * torch.ones_like(kwargs["y"])
        uncond_eps = model(x_t, ts, **uncond_kwargs)
        eps = uncond_eps + config.sampling.norm_cond_scale * (cond_eps - uncond_eps)
        return eps
    
    def inpainting_denoised_fn(x_start,**kwargs):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * kwargs['inpaint_mask']
            + kwargs['image'] * (1 - kwargs['inpaint_mask'])
    )

    def clamp_to_spatial_quantile(x : torch.Tensor, **kwargs):
        p = 0.99
        b, c, *spatial = x.shape
        quantile = torch.quantile(torch.abs(x).view(b,c,-1), p, dim = -1, keepdim =True)
        quantile = torch.max(quantile,torch.ones_like(quantile))
        quantile_broadcasted, _ = torch.broadcast_tensors(quantile.unsqueeze(-1),x)
        return torch.min(torch.max(x,-quantile_broadcasted), quantile_broadcasted) / quantile_broadcasted

    return cond_fn, model_fn, model_classifier_free_opt_fn, clamp_to_spatial_quantile



def get_dict_of_arrays(all_results):
    samples = {k: [get_numpy_from_torch(dic[k]) for dic in all_results] for k in all_results[0]}
    samples = {k:np.concatenate(v,0) for k,v in samples.items()}
    return samples

def get_numpy_from_torch(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        else:
            return tensor

def get_input_data(config, data_dict):

    model_kwargs = {k: v.to(dist_util.dev()) for k, v in data_dict.items()}
    model_kwargs["y"] = (config.sampling.target_class * torch.ones((config.sampling.batch_size,))).to(torch.long).to(dist_util.dev())
    
    init_image = data_dict['image'].to(dist_util.dev())

    return model_kwargs,init_image
