import torch
import os


def reload_device(device, model, task, device_idx):
    # device = 'cpu' or '0' or '0,1,2,3'
    if task == 'train':
        device = next(model.parameters()).device
    else:
        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif device:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            assert torch.cuda.is_available()
        cuda = device != 'cpu' and torch.cuda.is_available()
        device = torch.device('cuda:{}'.format(device_idx) if cuda else 'cpu')
    return device