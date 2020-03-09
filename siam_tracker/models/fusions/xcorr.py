# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Union, List

from .base_fusion import BaseFusion
from ..builder import FUSIONS
from ..utils import random_init_weights
from ...utils import center_crop


@FUSIONS.register_module
class CrossCorrelation(BaseFusion):

    def __init__(self,
                 feat_name: str,
                 in_channels: int,
                 corr_channels: int,
                 out_channels: Union[int, List],
                 depth_wise: bool,
                 pre_kernel_size: int,
                 linear_trans: int = -1,
                 share_pre_conv: bool = False,
                 init_type: str = None):
        """ Cross-correlation operation that widely used in SiamNet.
        Args:
            feat_name (str): which layer is used to perform cross-correlation.
            in_channels (int): number of input channels
            corr_channels (int): number of correlation channels. Because the outputs of backbone network
                                 usually end with a ReLU layer, which makes the features are not suitable
                                 for cross-correlation. Thus, we will firstly use a conv layer (without
                                 non-linear transform) to project the input features into another space.
                                 The conv kernel size is defined in 'pre_kernel_size' and the channels is
                                 defined in 'corr_channels'.
            out_channels (Union[int, List]): If out channels is a scalar integer, we will perform one cross-
                                             cross-correlation operation and return a Tensor. For some
                                             applications like SiamRPN, we need to perform two cross-correlation
                                             for two branches. The out_channels can be set to a list that denotes
                                             the output channels in two branches.
            depth_wise (bool): perform depth-wise cross-correlation or not. If True, the out_channels should be
                               same as corr_channels.
            pre_kernel_size (int): the kernel size in linear conv layer.
            linear_trans (int): For some backbones like ResNet-50, the number of channels are too large (e.g, 1024
                                or 2048). If we directly apply cross-correlation for this backbone, there will be
                                a large amount of parameters. Thus, we add a linear 1x1 conv layer to reduce the
                                number of channels. If this value is negative, it will be omitted.
            share_pre_conv (bool): whether share the preconv layer in template branch and search region branch.
            init_type (str): init type.
        """
        super(CrossCorrelation, self).__init__()
        self.feat_name = feat_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outputs = len(out_channels) if isinstance(out_channels, list) else 1

        self.depth_wise = depth_wise
        if linear_trans > 0:
            self.linear_proj = nn.Conv2d(in_channels, linear_trans, 1)
            in_channels = linear_trans
        else:
            self.linear_proj = None

        if self.num_outputs == 1:
            if depth_wise:
                z_corr_channels = corr_channels
                assert corr_channels == out_channels, \
                    'corr_channels == out_channels. ({} vs. {})'.format(corr_channels, out_channels)
            else:
                z_corr_channels = corr_channels * out_channels
            self.z_conv = nn.Conv2d(in_channels, z_corr_channels, pre_kernel_size)
            if not share_pre_conv:
                self.x_conv = nn.Conv2d(in_channels, corr_channels, pre_kernel_size)
            else:
                assert z_corr_channels == corr_channels
                self.x_conv = None
        else:
            assert not share_pre_conv
            z_conv_list = []
            x_conv_list = []
            for i in range(self.num_outputs):
                if depth_wise:
                    z_corr_channels = corr_channels
                    assert corr_channels == out_channels[i], \
                        'corr_channels == out_channels. ({} vs. {})'.format(corr_channels, out_channels[i])
                else:
                    z_corr_channels = corr_channels * out_channels[i]
                z_conv_list.append(nn.Conv2d(in_channels, z_corr_channels, pre_kernel_size))
                x_conv_list.append(nn.Conv2d(in_channels, corr_channels, pre_kernel_size))

            self.z_conv = nn.ModuleList(z_conv_list)
            self.x_conv = nn.ModuleList(x_conv_list)

        if init_type is not None:
            random_init_weights(self.modules(), init_type)

    def extract_z_feat(self, z_feats: Dict[str, Tensor], z_boxes: Tensor, cfg: Dict = None) -> Union[Tensor, List]:
        z_feat = z_feats[self.feat_name]
        if z_feat.size(3) > cfg.z_feat_size:
            # if the size is larger than a threshold, we will center crop the tensor
            z_feat = center_crop(z_feat, cfg.z_feat_size)
        if self.linear_proj is not None:
            z_feat = self.linear_proj(z_feat)
        # apply for the pre-convolution
        if self.num_outputs > 1:
            return [self.z_conv[i](z_feat) for i in range(self.num_outputs)]
        else:
            return self.z_conv(z_feat)

    def extract_x_feat(self, x_feats: Dict[str, Tensor], x_boxes: Tensor, cfg: Dict = None) -> Union[Tensor, List]:
        x_feat = x_feats[self.feat_name]
        if x_feat.size(3) > cfg.x_feat_size:
            # if the size is larger than a threshold, we will center crop the tensor
            x_feat = center_crop(x_feat, cfg.x_feat_size)
        if self.linear_proj is not None:
            x_feat = self.linear_proj(x_feat)
        # apply for the pre-convolution
        if self.num_outputs > 1:
            return [self.x_conv[i](x_feat) for i in range(self.num_outputs)]
        else:
            if self.x_conv is None:
                return self.z_conv(x_feat)
            else:
                return self.x_conv(x_feat)

    def fuse(self, z_feat, x_feat):
        if self.num_outputs > 1:
            return [self._fuse_single(z, x) for z, x in zip(z_feat, x_feat)]
        else:
            return self._fuse_single(z_feat, x_feat)

    def _fuse_single(self, z_feat, x_feat):
        """ Apply cross-correlation operation. """
        nx = x_feat.size(0)
        if z_feat.size(0) != nx:
            nz, cz, hz, wz = z_feat.size()
            assert nz == 1
            z_feat = z_feat.expand(nx, cz, hz, wz)
        if self.depth_wise:
            return xcorr_depthwise_op(x_feat, z_feat)
        else:
            return xcorr_op(x_feat, z_feat)


def xcorr_depthwise_op(x, kernel):
    """ depthwise cross correlation """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def xcorr_op(x, kernel):
    """ group conv2d to calculate cross correlation, fast version """
    batch = kernel.size(0)
    pk = kernel.view(-1, x.size(1), kernel.size(2), kernel.size(3))
    px = x.view(1, -1, x.size(2), x.size(3))
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size(2), po.size(3))
    return po
