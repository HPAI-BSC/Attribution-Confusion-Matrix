import os
import torch


def _get_mapping_key(state_dict):
    mapping_dict = {}
    for key in state_dict:
        mapping_dict[key] = ".".join(key.split('.')[1:])
    return mapping_dict

def _map_keys(state_dict):
    mapping_dict = _get_mapping_key(state_dict)
    all_keys = list(state_dict.keys())
    for key in all_keys:
        try:
            new_key = mapping_dict[key]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        except KeyError:
            continue
    return state_dict

def load_checkpoint(model_path, model, optimizer=None, epoch=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.isfile(model_path):
        print("=> no checkpoint found at '{}'".format(model_path))
        return
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(_map_keys(checkpoint['state_dict']))
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, epoch))
    return model, optimizer, epoch
