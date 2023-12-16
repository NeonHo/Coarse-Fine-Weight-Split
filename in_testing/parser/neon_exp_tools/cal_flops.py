# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from pathlib import Path
import torch

import torch.nn.functional as F
from torchstat import stat
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from mmcls.apis import init_model
from mmcls.models.classifiers import ImageClassifier
from in_testing.parser.neon_exp_tools.plot_tools import plot_hist
from mmcls.models.backbones import RepVGG
from mmcls.models.utils import SELayer
from mmcls.models.backbones.repvgg import RepVGGBlock


def visual_weight_after_before_conversion(model):
    print('Converting...')
    assert hasattr(model, 'backbone') and \
        hasattr(model.backbone, 'switch_to_deploy'), \
        '`model.backbone` must has method of "switch_to_deploy".' \
        f' But {model.backbone.__class__} does not have.'
    
    count = 0
    for m in model.backbone.modules():
        if isinstance(m, RepVGGBlock):
            weight_3x3, bias_3x3 = m._fuse_conv_bn(m.branch_3x3)
            weight_1x1, bias_1x1 = m._fuse_conv_bn(m.branch_1x1)
            # pad a conv1x1 weight to a conv3x3 weight
            weight_1x1 = F.pad(weight_1x1, [1, 1, 1, 1], value=0)

            weight_norm, bias_norm = 0, 0
            if m.branch_norm:
                tmp_conv_bn = m._norm_to_conv3x3(m.branch_norm)
                weight_norm, bias_norm = m._fuse_conv_bn(tmp_conv_bn)
                plot_hist(weight_norm, kernel_type="0", node_id=count)
    
            weight_converted, bias_converted = (weight_3x3 + weight_1x1 + weight_norm, bias_3x3 + bias_1x1 + bias_norm)
    
            plot_hist(weight_3x3.data, kernel_type="3", node_id=count)
            plot_hist(weight_1x1.data, kernel_type="1", node_id=count)
            plot_hist(weight_converted, kernel_type="4", node_id=count)
            
            count += 1


def main():
    parser = argparse.ArgumentParser(
        description='Convert the parameters of the repvgg block '
        'from training mode to deployment mode.')
    parser.add_argument(
        'config_path',
        help='The path to the configuration file of the network '
        'containing the repvgg block.')
    parser.add_argument(
        'checkpoint_path',
        help='The path to the checkpoint file corresponding to the model.')
    args = parser.parse_args()

    model = init_model(
        args.config_path, checkpoint=args.checkpoint_path, device='cpu')
    assert isinstance(model, ImageClassifier), \
        '`model` must be a `mmcls.classifiers.ImageClassifier` instance.'
    print('Converting...')
    assert hasattr(model, 'backbone') and \
        hasattr(model.backbone, 'switch_to_deploy'), \
        '`model.backbone` must has method of "switch_to_deploy".' \
        f' But {model.backbone.__class__} does not have.'
    model = RepVGG('A1', out_indices=(0, 1, 2, 3))
    model.switch_to_deploy()
    model.eval()
    stat(model, (3, 224, 224))
    

if __name__ == '__main__':
    main()
