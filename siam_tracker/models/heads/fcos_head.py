# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn.functional
from torch import nn
from typing import List, Union, Dict

from ..utils import build_stack_conv_layers, random_init_weights
from ..builder import HEADS, build_loss
from ...core import PointGenerator, generate_gt, assign_and_sample
from ...utils import multi_apply
from ...utils import box as ubox


@HEADS.register_module
class FCOSHead(nn.Module):

    def __init__(self,
                 stride: Union[List, int],
                 target_means: List = None,
                 target_stds: List = None,
                 pre_convs: Dict = None,
                 head_convs: List[Dict] = None,
                 cls_loss: Dict = None,
                 reg_loss: Dict = None,
                 multi_level_learnable_weights: bool = True,
                 init_type: str = 'xavier_uniform'):
        """ FCOS head is motivated by recent anchor-free object detector like FCOS [1] or
        FoveaBox [2]. They eliminate the design of anchors in RPN (or SiamRPN in tracking).
        The offsets between current pixel center and object boundary is directly predicted
        by the network. In tracking community, we have witnessed a trend to replace RPN
        head with this anchor-free head, like SiamFC++ [3] and SiamCARS [4].

        [1] fcos: fully convolutional one-stage object detection
        [2] FoveaBox: Beyond Anchor-based Object Detector
        [3] SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines
        [4] SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking

        """
        super(FCOSHead, self).__init__()

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

        self.point_gen = PointGenerator()
        self.cls_loss_obj = build_loss(cls_loss)
        self.reg_loss_obj = build_loss(reg_loss)

        if init_type is not None:
            random_init_weights(self.modules(), init_type)

    def forward(self, fused_feat):
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

    def get_boxes(self, cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, cfg: Dict) -> torch.Tensor:
        """ Convert the outputs to a series of bounding boxes (x1, y1, x2, y2).
        Args:
            cls_logits (torch.Tensor): in shape of [N, 1, H, W]
            bbox_deltas (torch.Tensor): in shape of [N, 4, H, W]
            cfg (dict): training or test configuration dict.
        Returns:
            boxes (torch.Tensor): in shape of [N, HW, 4]
        """
        img_ctr = (cfg.x_size - 1) / 2.0
        h, w = cls_logits.size()[-2:]  # [H, W]
        # generator center point list
        ctr_pts = self.point_gen.grid_points((h, w), stride=self.stride).type_as(cls_logits) + img_ctr
        boxes = self.ctrs_to_boxes(cls_logits, bbox_deltas, ctr_pts, self.target_means, self.target_stds)
        return boxes

    def loss(self, cls_logits, bbox_deltas, z_boxes, x_boxes, flags, cfg, **kwargs):
        """ Calculate the loss. """
        num_imgs = len(x_boxes)
        gt_boxes_list = generate_gt(z_boxes, x_boxes, flags, same_category_as_positive=False)
        img_ctr = (cfg.x_size - 1) / 2.0
        # generator center point list
        ctr_pts = self.point_gen.grid_points((cls_logits.size(2), cls_logits.size(3)),
                                             stride=self.stride).type_as(cls_logits) + img_ctr
        pseudo_boxes = ctr_pts.view(1, -1, 2).repeat(num_imgs, 1, 2)  # [N, HW, 4]

        # generate anchor targets
        cls_reg_targets = self.point_target(pseudo_boxes, gt_boxes_list, self.target_means, self.target_stds, cfg.fcos)
        (labels, label_weights, reg_targets, reg_weights, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg

        # calculate classification loss
        cls_logits = cls_logits.reshape(-1)
        cls_loss = self.cls_loss_obj(cls_logits, labels, label_weights, average_factor=num_total_samples)

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_loss = self.reg_loss_obj(bbox_deltas, reg_targets, reg_weights, average_factor=num_total_pos * 4)

        return dict(cls_loss=cls_loss, reg_loss=reg_loss)

    @staticmethod
    def point_target(points_list, gt_boxes_list, target_means, target_stds, cfg):
        (labels_list, label_weights_list, reg_targets_list, reg_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            FCOSHead.point_target_single,
            points_list,
            gt_boxes_list,
            target_means=target_means,
            target_stds=target_stds,
            cfg=cfg
        )
        # group into one
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        # group into one
        labels = torch.cat([_ for _ in labels_list], dim=0)
        label_weights = torch.cat([_ for _ in label_weights_list], dim=0)
        reg_targets = torch.cat([_ for _ in reg_targets_list], dim=0)
        reg_weights = torch.cat([_ for _ in reg_weights_list], dim=0)

        return labels, label_weights, reg_targets, reg_weights, num_total_pos, num_total_neg

    @staticmethod
    def point_target_single(points, gt_boxes, target_means, target_stds, cfg):
        num_pts = len(points)
        assign_result, sampling_result = assign_and_sample(points, gt_boxes, None, cfg)
        reg_targets = points.new_zeros(num_pts, 4)
        reg_weights = points.new_zeros(num_pts, 4)
        labels = points.new_zeros(num_pts, dtype=torch.long)
        label_weights = points.new_zeros(num_pts, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:

            pos_bbox_targets = ubox.corner2delta(sampling_result.pos_bboxes,
                                                 sampling_result.pos_gt_bboxes,
                                                 target_means, target_stds)
            reg_targets[pos_inds, :] = pos_bbox_targets
            if 'bbox_inner_weight' in cfg:
                pw = cfg.bbox_inner_weight
            else:
                pw = 1.0
            reg_weights[pos_inds, :] = pw
            labels[pos_inds] = 1
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, reg_targets, reg_weights, pos_inds, neg_inds

    @staticmethod
    def ctrs_to_boxes(cls_logits: torch.Tensor,
                      bbox_deltas: torch.Tensor,
                      ctr_pts: torch.Tensor,
                      target_means: List,
                      target_stds: List) -> torch.Tensor:
        """ Predict bboxes from network_output.
            Args:
                cls_logits (torch.Tensor): in shape of [N, 1, H, W],
                bbox_deltas (torch.Tensor): in shape of [N, 4, H, W],
                ctr_pts (torch.Tensor): in shape of [HW, 4].
                target_means (Iterable): Mean value of regression targets.
                target_stds (Iterable): Std value of regression targets.
            Returns:
                boxes (torch.Tensor): in shape of [N, HW, 5], the order in the last dimension is [x1, y1, x2, y2, score]
            """
        num_imgs = cls_logits.size(0)
        num_boxes = ctr_pts.size(0)
        cls_logits = cls_logits.view(num_imgs, num_boxes)
        cls_scores = cls_logits.sigmoid()

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(num_imgs, num_boxes, 4)
        boxes = cls_logits.new_zeros([num_imgs, num_boxes, 5])
        for i in range(num_imgs):
            i_boxes = ubox.delta2corner(ctr_pts, bbox_deltas[i], means=target_means, stds=target_stds)
            boxes[i, ...] = torch.cat((i_boxes, cls_scores[i].unsqueeze(-1)), dim=-1)
        return boxes
