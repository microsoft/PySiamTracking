# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch import nn
from collections import OrderedDict

from .base_backbone import BackboneCNN
from ..builder import BACKBONES


@BACKBONES.register_module
class AlexNet(BackboneCNN):
    """ AlexNet backbone that consists of 5 convolutional layers.
    Basically, the official PyTorch implementation of AlexNet [1] does not include BatchNorm.
    However, in tracking community, BN version of AlexNet is widely used. [2]. Both architecture
    is supported in our implementation, just by switching the BN flag 'use_bn'.

    [1] https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    [2] Fully-Convolutional Siamese Networks for Object Tracking, ECCVW 2016
    """

    num_blocks = 5
    blocks = dict(
        conv1=dict(stride=4, channel=96),
        conv2=dict(stride=8, channel=256),
        conv3=dict(stride=8, channel=384),
        conv4=dict(stride=8, channel=384),
        conv5=dict(stride=8, channel=256),
    )

    def __init__(self,
                 use_padding=True,
                 use_bn=True,
                 norm_eval=False,
                 pretrained=None,
                 init_type='xavier_uniform'):
        super(AlexNet, self).__init__()
        pad_pixs = [0 for i in range(7)]
        if use_padding:
            pad_pixs = [5, 1, 2, 1, 1, 1, 1]
        self.use_bn = use_bn
        self.norm_eval = norm_eval
        if use_bn:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(3, 64, 11, stride=2, padding=pad_pixs[0], bias=False)),
                ('bn', nn.BatchNorm2d(64)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=pad_pixs[1], dilation=1))]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(64, 192, 5, stride=1, padding=pad_pixs[2], groups=1, bias=False)),
                ('bn', nn.BatchNorm2d(192)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=pad_pixs[3], dilation=1))]))
            self.conv3 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(192, 384, kernel_size=3, padding=pad_pixs[4], bias=False)),
                ('bn', nn.BatchNorm2d(384)),
                ('relu', nn.ReLU())]))
            self.conv4 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 256, kernel_size=3, padding=pad_pixs[5], groups=1, bias=False)),
                ('bn', nn.BatchNorm2d(256)),
                ('relu', nn.ReLU())]))
            self.conv5 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(256, 256, kernel_size=3, padding=pad_pixs[6], groups=1, bias=False)),
                ('bn', nn.BatchNorm2d(256)),
                ('relu', nn.ReLU())]))
        else:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(3, 64, 11, stride=2, padding=pad_pixs[0], bias=True)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=pad_pixs[1], dilation=1))]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(64, 192, 5, stride=1, padding=pad_pixs[2], groups=1, bias=True)),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(3, stride=2, padding=pad_pixs[3], dilation=1))]))
            self.conv3 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(192, 384, kernel_size=3, padding=pad_pixs[4], bias=True)),
                ('relu', nn.ReLU())]))
            self.conv4 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(384, 256, kernel_size=3, padding=pad_pixs[5], groups=1, bias=True)),
                ('relu', nn.ReLU())]))
            self.conv5 = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(256, 256, kernel_size=3, padding=pad_pixs[6], groups=1, bias=True)),
                ('relu', nn.ReLU())]))

        self.init_weights(init_type, pretrained)
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train(self, mode=True):
        super(AlexNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
