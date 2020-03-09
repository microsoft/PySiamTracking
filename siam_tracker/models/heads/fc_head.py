# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn.functional
from torch import nn
from typing import List, Dict, Tuple

from ..utils import build_stack_conv_layers, random_init_weights
from ..builder import HEADS, build_loss
from ...core import PointGenerator, generate_gt, assign_and_sample
from ...utils import multi_apply


@HEADS.register_module
class FCHead(nn.Module):

    def __init__(self,
                 stride: int,
                 in_channels: int,
                 scale_factor: float,
                 head_convs: Dict = None,
                 loss: Dict = None,
                 init_type: str = 'xavier_uniform'):
        """ The head of SiamFC. It takes the fused features (or response map) as input and output the
        response map.

        Args:
            stride (int): feature stride of input feature maps
            in_channels (int): number of input channels,
            scale_factor (float): because the response map after cross-correlation is large, we scale them down
                                  by a scale factor (typically 0.001).
            head_convs (dict): head conv layers configurations
            loss (dict): classification loss configuration
            init_type (str): initialization type.

        """
        super(FCHead, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        if head_convs is not None:
            self.post_convs = build_stack_conv_layers(**head_convs)
        else:
            assert self.in_channels == 1
            self.post_convs = None
            self.score_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.point_gen = PointGenerator()
        self.cls_loss_obj = build_loss(loss)

        if init_type is not None:
            random_init_weights(self.modules(), init_type)

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """ Process the fused features (or response maps).
        Args:
            fused_feat (torch.Tensor): in shape of [N, C, H, W] (typically [1, 1, 17, 17])

        """
        if self.post_convs is not None:
            cls_logits = self.post_convs(fused_feat) * self.scale_factor
        else:
            cls_logits = fused_feat * self.scale_factor + self.score_bias
        return cls_logits

    def loss(self,
             cls_logits: torch.Tensor,
             z_boxes: List[torch.Tensor],
             x_boxes: List[torch.Tensor],
             flags: List[torch.Tensor],
             cfg: Dict,
             **kwargs) -> Dict[str, torch.Tensor]:
        """ Calculate the loss.

        Args:
            cls_logits (torch.Tensor): the predicted classification results, in shape of [N, 1, H, W]
            z_boxes (List[torch.Tensor]): the ground-truth boxes in template images. each element is in shape of [1, 6]
            x_boxes (List[torch.Tensor]): the ground-truth boxes in search regions. each element is in shape of [K, 6]
            flags (List[torch.Tensor]): bool tensors that denotes whether the search region and template are
                                        come from same sequence.
            cfg (dict): training configuration
        """
        num_imgs = len(x_boxes)
        # Generate ground-truth boxes
        gt_boxes_list = generate_gt(z_boxes, x_boxes, flags, same_category_as_positive=False)
        # Generate point coordinates for each pixel in the response map.
        img_ctr = (cfg.x_size - 1) / 2.0
        ctr_pts = self.point_gen.grid_points((cls_logits.size(2), cls_logits.size(3)),
                                             stride=self.stride).type_as(cls_logits) + img_ctr
        prior_boxes_list = [self.point_gen.assign_prior_boxes(ctr_pts, z_box[0:4].view(1, 4)) for z_box in z_boxes]
        # Generate ground-truth labels for each pixel.
        labels, label_weights = self.point_target(prior_boxes_list, gt_boxes_list, cfg.siamfc)
        cls_loss = self.cls_loss_obj(cls_logits.view(num_imgs, -1),
                                     labels.view(num_imgs, -1),
                                     label_weights.view(num_imgs, -1),
                                     average_factor=num_imgs)
        return dict(cls_loss=cls_loss)

    @staticmethod
    def point_target(prior_boxes_list: List[torch.Tensor],
                     gt_boxes_list: List[torch.Tensor],
                     cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate ground-truth target for each element.
        Args:
            prior_boxes_list (List[torch.Tensor]): each element in the list is a tensor in shape of [N, 4]
            gt_boxes_list (List[torch.Tensor]): ground-truth boxes list.
            cfg (dict): training configurations.
        """
        labels_list, label_weights_list = multi_apply(
            FCHead.point_target_single,
            prior_boxes_list,
            gt_boxes_list,
            cfg=cfg
        )
        # group into one
        labels = torch.cat([_ for _ in labels_list], dim=0)
        label_weights = torch.cat([_ for _ in label_weights_list], dim=0)

        return labels, label_weights

    @staticmethod
    def point_target_single(prior_boxes: torch.Tensor,
                            gt_boxes: torch.Tensor,
                            cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate ground-truth target for single image pair. """
        num_pts = len(prior_boxes)
        assign_result, sampling_result = assign_and_sample(prior_boxes, gt_boxes, None, cfg)
        labels = prior_boxes.new_zeros(num_pts).long()
        label_weights = prior_boxes.new_zeros(num_pts).float()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            labels[pos_inds] = 1
            if cfg.pos_weight > 0:
                sum_weight = 1.0 if len(neg_inds) <= 0 else cfg.pos_weight
                label_weights[pos_inds] = sum_weight / len(pos_inds)
        if len(neg_inds) > 0:
            if cfg.pos_weight > 0:
                sum_weight = 1.0 if len(pos_inds) <= 0 else 1 - cfg.pos_weight
                label_weights[neg_inds] = (sum_weight / len(neg_inds))

        total_samples = max(len(pos_inds) + len(neg_inds), 1)
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0 / total_samples
            label_weights[neg_inds] = 1.0 / total_samples

        return labels, label_weights


@HEADS.register_module
class RankFCHead(FCHead):

    def __init__(self,
                 stride: int,
                 in_channels: int,
                 scale_factor: float,
                 head_convs: List[int] = None,
                 head_ksize: int = None,
                 loss: Dict = None,
                 rank_loss: Dict = None,
                 init_type: str = 'xavier_uniform'):
        super(RankFCHead, self).__init__(stride, in_channels, scale_factor, head_convs, head_ksize, loss, init_type)
        self.rank_loss_obj = build_loss(rank_loss)

    def loss(self, cls_logits, z_boxes, x_boxes, flags, cfg, **kwargs):
        """ Calculate the loss. """
        num_imgs = len(x_boxes)
        gt_boxes_list = generate_gt(z_boxes, x_boxes, flags, same_category_as_positive=False)
        img_ctr = (cfg.x_size - 1) / 2.0
        # generator center point list
        ctr_pts = self.point_gen.grid_points((cls_logits.size(2), cls_logits.size(3)),
                                             stride=self.stride).type_as(cls_logits) + img_ctr
        prior_boxes_list = [self.point_gen.assign_prior_boxes(ctr_pts, z_box[0:4].view(1, 4)) for z_box in z_boxes]
        labels, label_weights, metrics = self.point_target(prior_boxes_list, gt_boxes_list, cfg.siamfc)
        cls_loss = self.cls_loss_obj(cls_logits.view(num_imgs, -1),
                                     labels.view(num_imgs, -1),
                                     label_weights.view(num_imgs, -1),
                                     average_factor=num_imgs)
        losses = dict(cls_loss=cls_loss)
        if self.rank_loss_obj is not None:
            losses['rank_loss'] = self.rank_loss_obj(cls_logits.view(num_imgs, -1),
                                                     labels.view(num_imgs, -1),
                                                     metrics.view(num_imgs, -1))
        return losses

    @staticmethod
    def point_target(prior_boxes_list, gt_boxes_list, cfg):
        labels_list, label_weights_list, metrics_list = multi_apply(
            FCHead.point_target_single,
            prior_boxes_list,
            gt_boxes_list,
            cfg=cfg
        )
        # group into one
        labels = torch.cat([_ for _ in labels_list], dim=0)
        label_weights = torch.cat([_ for _ in label_weights_list], dim=0)
        metrics = torch.cat([_ for _ in metrics_list], dim=0)

        return labels, label_weights, metrics

    @staticmethod
    def point_target_single(prior_boxes, gt_boxes, cfg):
        num_pts = len(prior_boxes)
        assign_result, sampling_result = assign_and_sample(prior_boxes, gt_boxes, None, cfg)
        labels = prior_boxes.new_zeros(num_pts).long()
        label_weights = prior_boxes.new_zeros(num_pts).float()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        metrics = prior_boxes.new_zeros(num_pts)
        if len(pos_inds) > 0:
            labels[pos_inds] = 1
            metrics[pos_inds] = assign_result.max_overlaps[pos_inds]
            if cfg.pos_weight > 0:
                sum_weight = 1.0 if len(neg_inds) <= 0 else cfg.pos_weight
                label_weights[pos_inds] = sum_weight / len(pos_inds)
        if len(neg_inds) > 0:
            if cfg.pos_weight > 0:
                sum_weight = 1.0 if len(pos_inds) <= 0 else 1 - cfg.pos_weight
                label_weights[neg_inds] = (sum_weight / len(neg_inds))

        total_samples = max(len(pos_inds) + len(neg_inds), 1)
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0 / total_samples
            label_weights[neg_inds] = 1.0 / total_samples

        return labels, label_weights, metrics
