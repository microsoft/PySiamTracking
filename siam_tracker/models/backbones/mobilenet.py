# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch import nn
from .base_backbone import BackboneCNN

from ..builder import BACKBONES


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=True):
        if padding:
            padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, padding=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, padding=padding))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, padding=padding),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(BackboneCNN):
    def __init__(self,
                 num_stages=8,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 norm_eval=True,
                 pretrained=None,
                 init_type='xavier_uniform'):
        """
        MobileNet V2 main class. This class is taken from official PyTorch repo:
        https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py

        Args:
            num_stages (int): how many stages used in backbone
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number.
                                 Set to 1 to turn off rounding
            norm_eval (bool): If set to True, the running mean & var in BN layer will be frozen durning training.
            pretrained (str): pretrained model path
            init_type (str): which kind of initializer is used in backbone.
        """
        super(MobileNetV2, self).__init__()
        self.num_blocks = num_stages

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 1],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
        inverted_residual_setting = inverted_residual_setting[:num_stages-1]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        self.conv1 = ConvBNReLU(3, input_channel, stride=2, padding=False)
        self.blocks['conv1'] = dict(stride=2, channel=input_channel)

        stage_count = 1
        stage_stride = 2
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            features = []
            for i in range(n):
                stride = s if i == 0 else 1
                padding = not (i == 0 and stage_count == 1)
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, padding=padding))
                input_channel = output_channel
            stage_count = stage_count + 1
            stage_stride = stage_stride * s
            setattr(self, 'conv{}'.format(stage_count), nn.Sequential(*features))
            self.blocks['conv{}'.format(stage_count)] = dict(stride=stage_stride, channel=output_channel)
        self.init_weights(init_type, pretrained)

        self.norm_eval = norm_eval
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
