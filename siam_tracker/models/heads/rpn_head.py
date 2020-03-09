# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn.functional
from torch import nn
from typing import List, Union, Dict, Tuple

from ..utils import build_stack_conv_layers, random_init_weights
from ..builder import HEADS, build_loss
from ...core import AnchorGenerator, generate_gt, bbox_target
from ...utils import box as ubox


@HEADS.register_module
class RPNHead(nn.Module):

    def __init__(self,
                 stride: Union[int, List] = None,
                 anchor_scales: List = None,
                 anchor_ratios: List = None,
                 target_means: List = None,
                 target_stds: List = None,
                 pre_convs: Dict = None,
                 head_convs: List[Dict] = None,
                 cls_loss: Dict = None,
                 reg_loss: Dict = None,
                 multi_level_learnable_weights: bool = False,
                 init_type: str = 'xavier_uniform'):
        """ SiamRPN head module.
        Args:
            stride: the feature stride.
            anchor_scales: anchor scale configurations, typically [8.0, ],
            anchor_ratios: anchor ratio configurations
            target_means: we will subtract the mean values of ground-truth targets.
            target_stds: similar to target_means, we divide the std of ground-truth targets.
            pre_convs: before applying the final classification & regression head,
                       we will firstly adopt some conv layers (usually for depth-wise xcorr case).
            head_convs: how many conv layers used in two branches.
            cls_loss: classification loss configuration
            reg_loss: regression loss configuration
            multi_level_learnable_weights: enable learnable weights for multi-level outputs.
            init_type: initialization configuration
        """
        super(RPNHead, self).__init__()

        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.target_means = target_means
        self.target_stds = target_stds

        self.stride = stride
        # if stride is a a list or tuple, it means that we use multi-level RPN head which
        # is introduced by SiamRPN++. The outputs will be aggregated by learnable weights or
        # just simple average.
        if isinstance(self.stride, (tuple, list)):
            for s in self.stride:
                assert s == self.stride[0], "All elements in strides should be same."
            self.num_levels = len(self.stride)
            assert self.num_levels > 1
            self.stride = self.stride[0]
        else:
            self.num_levels = 1

        if pre_convs is not None:
            if self.num_levels > 1:
                self.pre_convs = nn.ModuleList(
                    [build_stack_conv_layers(**pre_convs[i]) for i in range(self.num_levels)]
                )
            else:
                self.pre_convs = build_stack_conv_layers(**pre_convs)
        else:
            self.pre_convs = None

        if self.num_levels > 1:
            self.cls_convs = nn.ModuleList(
                [build_stack_conv_layers(**head_convs[i][0]) for i in range(self.num_levels)]
            )
            self.reg_convs = nn.ModuleList(
                [build_stack_conv_layers(**head_convs[i][1]) for i in range(self.num_levels)]
            )
            self.cls_level_weights = nn.Parameter(torch.ones(self.num_levels, 1, 1, 1, 1),
                                                  requires_grad=False)
            self.reg_level_weights = nn.Parameter(torch.ones(self.num_levels, 1, 1, 1, 1),
                                                  requires_grad=False)
            if multi_level_learnable_weights:
                self.cls_level_weights.requires_grad_(True)
                self.reg_level_weights.requires_grad_(True)

        else:
            assert isinstance(head_convs, list) and len(head_convs) == 2
            self.cls_convs = None
            self.reg_convs = None
            if head_convs[0] is not None:
                self.cls_convs = build_stack_conv_layers(**head_convs[0])
            if head_convs[1] is not None:
                self.reg_convs = build_stack_conv_layers(**head_convs[1])

        self.anchor_gen = AnchorGenerator(base_size=self.stride,
                                          scales=self.anchor_scales,
                                          ratios=self.anchor_ratios,
                                          ctr=(0, 0))
        self.anchors_cached = None
        self.cls_loss_obj = build_loss(cls_loss)
        self.reg_loss_obj = build_loss(reg_loss)

        if init_type is not None:
            random_init_weights(self.modules(), init_type)

    def forward(self, fused_feat: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ predict the classification score and regression box deltas. """

        if self.pre_convs is not None:
            if self.num_levels > 1:
                fused_feat = [self.pre_convs[i](fused_feat[i]) for i in range(self.num_levels)]
            else:
                fused_feat = self.pre_convs(fused_feat)

        if self.num_levels > 1:
            if isinstance(fused_feat[0], (list, tuple)):
                cls_logits = [self.cls_convs[i](fused_feat[i][0]).unsqueeze(0) for i in range(self.num_levels)]
                bbox_deltas = [self.reg_convs[i](fused_feat[i][1]).unsqueeze(0) for i in range(self.num_levels)]
            else:
                cls_logits = [self.cls_convs[i](fused_feat[i]).unsqueeze(0) for i in range(self.num_levels)]
                bbox_deltas = [self.reg_convs[i](fused_feat[i]).unsqueeze(0) for i in range(self.num_levels)]
            # merge the results from different levels
            cls_weight = torch.nn.functional.softmax(self.cls_level_weights, 0)
            reg_weight = torch.nn.functional.softmax(self.reg_level_weights, 0)
            cls_logits = (torch.cat(cls_logits, dim=0) * cls_weight).sum(dim=0)
            bbox_deltas = (torch.cat(bbox_deltas, dim=0) * reg_weight).sum(dim=0)

        else:
            if isinstance(fused_feat, (list, tuple)):
                if self.cls_convs is not None:
                    cls_logits = self.cls_convs(fused_feat[0])
                else:
                    cls_logits = fused_feat[0]
                if self.reg_convs is not None:
                    bbox_deltas = self.reg_convs(fused_feat[1])
                else:
                    bbox_deltas = fused_feat[1]
            else:
                cls_logits = self.cls_convs(fused_feat)
                bbox_deltas = self.reg_convs(fused_feat)

        return cls_logits, bbox_deltas

    def get_anchors(self, featmap_size: Tuple[int, int], x_size: int) -> torch.Tensor:
        """ Generate anchor boxes according to feat map size.
        Assume that featmap_size is [h, w], the returned Tensor will be in shape of [h*w, 4], where 4
        dimensions denote (x1, y2, x2, y2).
        """
        if self.anchors_cached is not None:
            expected_num = featmap_size[0] * featmap_size[1] * len(self.anchor_scales) * len(self.anchor_ratios)
            if expected_num == self.anchors_cached.size(0):
                return self.anchors_cached
        # generate anchors.
        anchors = self.anchor_gen.grid_anchors(featmap_size, self.stride, 'cpu')
        search_region_center = (x_size - 1.0) / 2.0

        anchors = anchors + search_region_center  # align center point.
        self.anchors_cached = anchors
        return anchors

    def loss(self,
             cls_logits: torch.Tensor,
             bbox_deltas: torch.Tensor,
             z_boxes: List[torch.Tensor],
             x_boxes: List[torch.Tensor],
             flags: torch.Tensor,
             cfg: Dict,
             **kwargs) -> Dict[str, torch.Tensor]:
        """ Calculate the loss.

        Args:
            cls_logits (torch.Tensor): the predicted classification results, in shape of [N, 2A, H, W]
            bbox_deltas (torch.Tensor): the predicted bbox delta results, in shape of [N, 4A, H, W]
            z_boxes (List[torch.Tensor]): the ground-truth boxes in template images. each element is in shape of [1, 6]
            x_boxes (List[torch.Tensor]): the ground-truth boxes in search regions. each element is in shape of [K, 6]
            flags (torch.Tensor): bool tensors that denotes whether the search region and template are
                                        come from same sequence.
            cfg (dict): training configuration
        """
        num_imgs = len(x_boxes)
        gt_boxes_list = generate_gt(z_boxes, x_boxes, flags, same_category_as_positive=False)
        # generator anchor list
        featmap_size = cls_logits.size()[-2:]  # [H, W]
        anchors = self.get_anchors(featmap_size, cfg.x_size).type_as(cls_logits)
        anchors_list = [anchors for i in range(num_imgs)]
        # generate anchor targets
        cls_reg_targets = bbox_target(anchors_list, gt_boxes_list, self.target_means, self.target_stds, cfg.rpn)
        (labels, label_weights, bbox_targets, bbox_weights, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg

        # calculate classification loss
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(-1, 2)
        cls_loss = self.cls_loss_obj(cls_logits, labels, label_weights, average_factor=num_total_samples)
        # calculate box-regression loss
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_loss = self.reg_loss_obj(bbox_deltas, bbox_targets, bbox_weights, average_factor=num_total_pos)

        return dict(cls_loss=cls_loss, reg_loss=reg_loss)

    def get_boxes(self, cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, cfg: Dict) -> torch.Tensor:
        """ Convert the outputs to a series of bounding boxes (x1, y1, x2, y2).
        Args:
            cls_logits (torch.Tensor): in shape of [N, 2A, H, W]
            bbox_deltas (torch.Tensor): in shape of [N, 4A, H, W]
            cfg (dict): training or test configuration dict.
        Returns:
            boxes (torch.Tensor): in shape of [N, HWA, 4]
        """
        featmap_size = cls_logits.size()[-2:]  # [H, W]
        anchors = self.get_anchors(featmap_size, cfg['x_size']).type_as(cls_logits)
        boxes = self.anchor_to_boxes(cls_logits, bbox_deltas, anchors, self.target_means, self.target_stds)
        return boxes

    @staticmethod
    def anchor_to_boxes(cls_logits: torch.Tensor,
                        bbox_deltas: torch.Tensor,
                        anchors: torch.Tensor,
                        target_means: List,
                        target_stds: List) -> torch.Tensor:
        """ Predict bboxes from RPN.
            Args:
                cls_logits (torch.Tensor): in shape of [N, 2A, H, W],
                bbox_deltas (torch.Tensor): in shape of [N, 4A, H, W],
                anchors (torch.Tensor): in shape of [HWA, 4] where A is the number of anchors in echo pixel.
                target_means (Iterable): Mean value of regression targets.
                target_stds (Iterable): Std value of regression targets.

            Returns:
                boxes (torch.Tensor): in shape of [N, HWA, 5], the order in the last dimension is [x1, y1, x2, y2, score]
            """
        num_imgs = cls_logits.size(0)
        num_anchors = anchors.size(0)

        cls_logit = cls_logits.permute(0, 2, 3, 1).contiguous().view(num_imgs, num_anchors, 2)
        # 1 for foreground & 0 for background
        cls_scores = torch.nn.functional.softmax(cls_logit, dim=-1)[:, :, 1]
        bbox_delta = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(num_imgs, num_anchors, 4)
        boxes = cls_logit.new_zeros([num_imgs, num_anchors, 5])
        # TODO(guangting), replace loop with broadcast operation?.
        for i in range(num_imgs):
            i_boxes = ubox.delta2bbox(anchors, bbox_delta[i], means=target_means, stds=target_stds)
            boxes[i, ...] = torch.cat((i_boxes, cls_scores[i].unsqueeze(-1)), dim=-1)
        return boxes
