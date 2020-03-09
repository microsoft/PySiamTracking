# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.


import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BackboneCNN
from ..utils import build_conv_layer, build_norm_layer, random_init_weights, constant_init


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 normalize=dict(type='BN')):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_cfg=None,
                 normalize=dict(type='BN')):
        """Bottleneck block for ResNet.
        """
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv_cfg = conv_cfg
        self.normalize = normalize
        self.conv1_stride = 1
        self.conv2_stride = stride


        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            normalize, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.normalize = normalize

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   conv_cfg=None,
                   normalize=dict(type='BN')):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            conv_cfg=conv_cfg,
            normalize=normalize))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                conv_cfg=conv_cfg,
                normalize=normalize))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(BackboneCNN):
    """ResNet backbone. This class is taken from MMDetection repo:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    blocks = dict(
        conv1=dict(stride=4, channel=64),
    )

    def __init__(self,
                 depth,
                 num_stages=5,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 stem_padding=3,
                 conv_cfg=None,
                 normalize=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 zero_init_residual=True,
                 pretrained=None,
                 init_type='xavier_uniform'):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert 2 <= num_stages <= 5
        self.num_blocks = num_stages
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages - 1
        self.conv_cfg = conv_cfg
        self.normalize = normalize
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block_type, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages-1]
        self.inplanes = 64

        self.conv1 = self._make_stem_layer(stem_padding)  # we treat stem layers as 'conv1'

        _total_stride = 4  # conv 1 is stride 4
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block_type,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                normalize=normalize
            )
            self.inplanes = planes * self.block_type.expansion
            layer_name = 'conv{}'.format(i + 2)
            self.add_module(layer_name, res_layer)

            _total_stride = _total_stride * stride
            # TODO(guangting): add receptive field attribute & size function.
            self.blocks[layer_name] = dict(stride=_total_stride, channel=self.inplanes)

        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        self.init_weights(init_type, pretrained)

    def _make_stem_layer(self, stem_padding):
        layers = [build_conv_layer(self.conv_cfg, 3, 64, kernel_size=7, stride=2, padding=stem_padding, bias=False)]
        norm1_name, norm1 = build_norm_layer(self.normalize, 64, postfix=1)
        layers.append(norm1)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def init_weights(self, init_type='xavier_uniform', pretrained=None):
        if isinstance(pretrained, str):
            self.init_weights_from_pretrained(pretrained)
        elif pretrained is None:
            random_init_weights(self.modules(), init_type)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
