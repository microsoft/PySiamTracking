# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch import nn
from typing import List, Union, Dict

from ..builder import NECKS
from ..utils import build_stack_conv_layers, random_init_weights


@NECKS.register_module
class Adjust(nn.Module):

    def __init__(self,
                 feat_names: Union[str, List],
                 in_channels: Union[str, List],
                 out_channels: Union[str, List],
                 num_layers: Union[int, List],
                 kernel_size: Union[int, List],
                 init_type: str = None,
                 **kwargs):
        super(Adjust, self).__init__()
        if isinstance(feat_names, str):
            feat_names = [feat_names]
        self.feat_names = feat_names
        num_levels = len(self.feat_names)
        self.num_levels = num_levels

        if not isinstance(in_channels, (tuple, list)):
            in_channels = [in_channels for _ in range(num_levels)]
        if not isinstance(out_channels, (tuple, list)):
            out_channels = [out_channels for _ in range(num_levels)]
        if not isinstance(num_layers, (tuple, list)):
            num_layers = [num_layers for _ in range(num_levels)]
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = [kernel_size for _ in range(num_levels)]

        adjust_modules = []
        for i in range(num_levels):
            adjust_modules.append(
                build_stack_conv_layers(num_layers=num_layers[i],
                                        in_channels=in_channels[i],
                                        out_channels=out_channels[i],
                                        kernel_size=kernel_size[i],
                                        **kwargs)
            )
        self.adjust_modules = nn.ModuleList(adjust_modules)
        random_init_weights(self.modules(), init_type)

    def forward(self, feats: Dict[str, torch.Tensor]):
        for i in range(self.num_levels):
            feat_name = self.feat_names[i]
            feats[feat_name] = self.adjust_modules[i](feats[feat_name])
        return feats
