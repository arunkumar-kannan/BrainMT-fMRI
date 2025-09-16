import torch
from torch import optim as optim
import json

try:
    from .distributed import get_rank
except:
    def get_rank():
        return 0



def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith("transformer.resblocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif var_name in ("class_embedding", "positional_embedding", "temporal_positional_embedding"):
        return 0
    elif var_name.startswith("conv1"):
        return 0
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layer=None, 
        get_layer_scale=None,
    ):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def create_optimizer(
        model, get_num_layer=None, get_layer_scale=None, 
        filter_bias_and_bn=True, skip_list=None
    ):
    opt_lower = 'adamw'
    weight_decay = 0.05
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.module.no_weight_decay()
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale,
        )
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        pass

    opt_args = dict(lr=2e-4, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-6)

    if get_rank() == 0:
        print("optimizer settings:", opt_args)
    optimizer = optim.AdamW(parameters, **opt_args)

    return optimizer
