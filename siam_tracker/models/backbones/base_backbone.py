# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import logging
from torch import nn, Tensor
from collections import OrderedDict

from typing import Union, List, Dict

from ..utils import random_init_weights
from ...utils import load_checkpoint


class BackboneCNN(nn.Module):

    num_blocks = 0
    blocks = dict()

    def __init__(self):
        """ BackboneCNN is the base class for all backbone networks in siamese network, e.g.
        AlexNet or ResNet. It provides some useful utilization functions like initialization
        network parameters, fix some layers during training and so on.

        In our implementation, the different blocks in CNN are named as 'conv1', 'conv2', ...
        It provides an easier interface to control the network.
        """
        super(BackboneCNN, self).__init__()

    @classmethod
    def infer_channels(cls, tensor_name: Union[str, List[str]]) -> Union[int, List[int]]:
        """ Get the number of channels according to the tensor name (like 'conv1', 'conv2', ... )"""
        if isinstance(tensor_name, (tuple, list)):
            return [cls.infer_channels(tn) for tn in tensor_name]
        assert isinstance(tensor_name, str), "Unknown type of tensor name: [{}]".format(type(tensor_name))
        return cls.blocks[tensor_name]['channel']

    @classmethod
    def infer_stride(cls, tensor_name: Union[str, List[str]]) -> Union[int, List[int]]:
        """ Get the feature strides according to the tensor name (like 'conv1', 'conv2', ... )"""
        if isinstance(tensor_name, (tuple, list)):
            return [cls.infer_stride(tn) for tn in tensor_name]
        assert isinstance(tensor_name, str), "Unknown type of tensor name: [{}]".format(type(tensor_name))
        return cls.blocks[tensor_name]['stride']

    def init_weights_from_pretrained(self, pretrained: str) -> None:
        """ Loading pretrained model.
        Args:
            pretrained (str): 'imagenet' or file path
        """
        logger = logging.getLogger()
        logger.info("Loading pretrained model from {}".format(pretrained))
        load_checkpoint(self, pretrained, strict=False, logger=logger)

    def init_weights(self, init_type: str = 'xavier_uniform', pretrained: str = None) -> None:
        random_init_weights(self.modules(), init_type)
        if pretrained is not None:
            self.init_weights_from_pretrained(pretrained)

    def freeze_block(self, num_blocks):
        """ Freeze the parameters in the target blocks """
        logger = logging.getLogger()
        logger.info("Freeze backbone {} blocks...".format(num_blocks))
        for i in range(num_blocks):
            sub_module = getattr(self, 'conv{}'.format(i+1))
            for name, m in sub_module.named_modules():
                if isinstance(m, torch.nn.Conv2d):
                    logger.info("Freeze conv layer {} [{}]".format('conv{}'.format(i+1), name))
                    m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    logger.info("Freeze BN layer {} [{}]".format('conv{}'.format(i+1), name))
                    if m.weight is not None:
                        m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False
                    m.eval()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """ Forward the image through the backbone network.
        Basically, the network is divided into several stages, namely 'conv1', 'conv2', ...
        This function will return the extracted feature maps in each conv layers. For example,
        the input x is a tensor in shape of [N, 3, 255, 255]. The output dictionary may be
        {'conv1': Tensor[N, C, 64, 64], 'conv2': Tensor[N, C, 32, 32], ...}
        """
        out = OrderedDict()
        for i in range(1, self.num_blocks+1):
            op_name = 'conv{}'.format(i)
            op = getattr(self, op_name)
            if i == 1:
                out['conv1'] = op(x)
            else:
                out[op_name] = op(out['conv{}'.format(i-1)])
        return out
