# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn as nn


def build_stack_fc_layers(num_layers,
                          in_channels,
                          out_channels,
                          nonlinear=True,
                          nonlinear_last=True):
    layers = []
    if isinstance(out_channels, int):
        out_channels = [out_channels for i in range(num_layers)]

    if isinstance(nonlinear, bool):
        nonlinear = [nonlinear for i in range(num_layers)]
    nonlinear[-1] = nonlinear_last
    for i in range(num_layers):

        layers.append(nn.Linear(in_channels, out_channels[i]))
        if nonlinear[i]:
            layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels[i]
    layers = nn.Sequential(*layers)
    return layers
