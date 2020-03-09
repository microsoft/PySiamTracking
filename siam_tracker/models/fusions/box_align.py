# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch import nn, Tensor
from typing import Dict, Union, List

from .base_fusion import BaseFusion
from ..builder import FUSIONS
from ..utils import random_init_weights, build_stack_conv_layers
from ...ops.roi_align import RoIAlign


@FUSIONS.register_module
class BoxAlign(BaseFusion):

    def __init__(self,
                 feat_name: Union[str, List],
                 in_channels: Union[int, List],
                 stride: Union[int, List],
                 out_size: int,
                 post_convs: Dict = None,
                 fuse_type: str = 'concat',
                 sample_num: int = 2,
                 init_type: str = 'xavier_uniform'):
        """ Fuse the regional features w.r.t. the box coordinates. RoI Align operation is adopted to
        extract the regional features from template and search region.

        Args:
            feat_name (str or List[str]): the source layer name of feature maps. If 'feat_name' is a list,
                                     the feats are extracted from multi-layers.
            in_channels (int or List[int]): the number of input channels
            stride (int ro List[int]): the feature stride (typically 8)
            out_size (int): output feature map size. (the size of RoI align outputs)
            post_convs (Dict): the conv layer configuration after feature concat.
            fuse_type (str): which kind of op to fuse the template features and search region features.
            sample_num (int): sample number in RoI Align
            init_type (str): initialization type.

        """
        super(BoxAlign, self).__init__()
        self.feat_name = feat_name
        self.in_channels = in_channels
        out_channels = in_channels
        self.stride = stride
        if isinstance(feat_name, (tuple, list)):
            if not isinstance(in_channels, (tuple, list)):
                self.in_channels = [int(in_channels) for _ in feat_name]
            if not isinstance(stride, (list, tuple)):
                self.stride = [float(stride) for _ in feat_name]
            out_channels = sum(out_channels)

        self.out_size = out_size
        if isinstance(self.stride, (tuple, list)):
            self.roi_op = nn.ModuleList(
                [RoIAlign(out_size, 1.0 / st, sample_num=sample_num) for st in self.stride]
            )
        else:
            self.roi_op = RoIAlign(out_size, 1.0 / self.stride, sample_num=sample_num)

        assert fuse_type in ('concat', 'sum', 'diff')
        self.fuse_type = fuse_type
        if fuse_type == 'concat':
            out_channels = out_channels * 2

        if post_convs is not None:
            self.post_convs = build_stack_conv_layers(**post_convs)
        else:
            self.post_convs = None

        if init_type is not None:
            random_init_weights(self.modules(), init_type)

    def forward(self,
                z_feats: Dict[str, Tensor] = None,
                x_feats: Dict[str, Tensor] = None,
                z_info: List[Tensor] = None,
                x_info: List[Tensor] = None,
                cfg: Dict = None) -> Tensor:
        if z_feats is None and x_feats is not None:
            # load template features from cache
            x_feat = self.extract_x_feat(x_feats, x_info, cfg)
            return self.fuse(self.z_cache, x_feat)
        elif x_feats is None and z_feats is not None:
            self.z_cache = self.extract_z_feat(z_feats, z_info, cfg)
        elif z_feats is not None and x_feats is not None:
            z_info = [z_box.view(1, -1).repeat(x_box.size(0), 1) for z_box, x_box in zip(z_info, x_info)]
            z_feat = self.extract_z_feat(z_feats, z_info, cfg)
            x_feat = self.extract_x_feat(x_feats, x_info, cfg)
            return self.fuse(z_feat, x_feat)
        else:
            raise ValueError("At least one element of z_feats and x_feats should be NOT NONE.")

    def extract_z_feat(self, z_feats: Dict[str, Tensor], z_boxes: List[Tensor], cfg: Dict):
        z_roi_feats = self._extract(z_feats, z_boxes, cfg['x_size'])
        return z_roi_feats

    def extract_x_feat(self, x_feats: Dict[str, Tensor], x_boxes: List[Tensor], cfg: Dict):
        x_roi_feats = self._extract(x_feats, x_boxes, cfg['x_size'])
        return x_roi_feats

    def fuse(self, z_feat, x_feat):
        if z_feat.size(0) == 1 and x_feat.size(0) > 1:
            z_feat = z_feat.expand_as(x_feat)

        if self.fuse_type == 'concat':
            fuse_feats = torch.cat((z_feat, x_feat), dim=1)
        elif self.fuse_type == 'sum':
            fuse_feats = z_feat + x_feat
        elif self.fuse_type == 'diff':
            fuse_feats = z_feat - x_feat
        else:
            raise ValueError
        if self.post_convs is not None:
            fuse_feats = self.post_convs(fuse_feats)
        return fuse_feats

    def _extract(self, feats: Dict[str, Tensor], boxes: List[Tensor], x_size: int) -> Tensor:
        """ Apply RoI Align in each level of feature maps.
        Args:
            feats (dict): the key is 'conv1', 'conv2', ...
            boxes (List[Tensor]): each element is a [M, 4] tensor that indicates the coordinates of
                                  (x1, y1, x2, y2).
            x_size (int): input image size.
        """
        rois = self.group_rois(boxes)
        img_ctr = (x_size - 1.) / 2.
        # apply roi align op
        if isinstance(self.feat_name, (list, tuple)):
            # multi-level features
            roi_feats = []
            for i in range(len(self.feat_name)):
                lvl_rois = rois.clone()
                feat_ctr = (feats[self.feat_name[i]].size(2) - 1.) / 2.
                lvl_rois[:, 1:5] += (feat_ctr * self.stride[i] - img_ctr)
                roi_feats.append(self.roi_op[i](feats[self.feat_name[i]], lvl_rois))
            roi_feats = torch.cat(roi_feats, dim=1)
        else:
            # single-level features
            feat_ctr = (feats[self.feat_name].size(2) - 1.) / 2.
            rois[:, 1:5] += (feat_ctr * self.stride - img_ctr)
            roi_feats = self.roi_op(feats[self.feat_name], rois)
        return roi_feats

    @staticmethod
    def group_rois(boxes: List[Tensor]) -> Tensor:
        """ Group a list of boxes into a single tensor. Note that RoI Align Op requires the
        input to be something like [[batch_idx, x1, y1, x2, y2], ...]. """
        if isinstance(boxes, Tensor):
            assert boxes.size(-1) == 4
            if boxes.dim() == 2:
                roi_inds = boxes.new_zeros((boxes.size(0), 1))
                rois = torch.cat((roi_inds, boxes), dim=1)
            elif boxes.dim() == 3:
                num_img = boxes.size(0)
                num_roi_per_img = boxes.size(1)  # [N, K, 4]
                roi_inds = torch.arange(num_img).view(-1, 1).repeat(1, num_roi_per_img).type_as(boxes)
                rois = torch.cat((roi_inds.view(-1, 1), boxes.view(-1, 4)), dim=1)  # [NK, 5]
            else:
                raise ValueError("unsupport type {} and shape {}".format(type(boxes), boxes.size()))
        else:
            rois = []
            for i, box in enumerate(boxes):
                i_inds = box.new_zeros((box.size(0), 1)) + i
                rois.append(torch.cat((i_inds, box), dim=1))  # [K, 5]
            rois = torch.cat(rois, dim=0)  # [NK, 5]
        return rois
