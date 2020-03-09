# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn.functional
from torch import nn
from typing import List, Dict, Tuple

from ..builder import HEADS, build_loss
from ..utils import build_stack_fc_layers, random_init_weights
from ...core import generate_gt, bbox_target
from ...utils import box as ubox


@HEADS.register_module
class CompareHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 in_size: int,
                 target_means: List = None,
                 target_stds: List = None,
                 num_fcs: int = 2,
                 feat_channels: int = 256,
                 cls_loss: Dict = None,
                 reg_loss: Dict = None,
                 init_type: str = 'xavier_uniform'):
        """ Compare the template and search region features by a fully connected
        network. It severs the fine-matching (FM) stage in SPM tracker [1], and firstly
        introduced in Relation Network [2]

        [1] SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking
        [2] Learning to compare relation network for few-shot learning
        """
        super(CompareHead, self).__init__()
        self.target_means = target_means
        self.target_stds = target_stds
        in_channels = in_channels * in_size * in_size
        self.feat_fcs = build_stack_fc_layers(num_fcs, in_channels, feat_channels, nonlinear_last=True)
        self.cls_fc = nn.Linear(feat_channels, 2, bias=True)
        self.reg_fc = nn.Linear(feat_channels, 4, bias=True)

        self.cls_loss_obj = build_loss(cls_loss)
        self.reg_loss_obj = build_loss(reg_loss)

        if init_type is not None:
            random_init_weights(self.modules(), init_type)

    def forward(self, fused_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate the similarity score and relative box distance. """
        num = fused_feat.size(0)
        fused_feat = fused_feat.view(num, -1)
        fused_feat = self.feat_fcs(fused_feat)
        cls_logits = self.cls_fc(fused_feat)
        bbox_deltas = self.reg_fc(fused_feat)
        return cls_logits, bbox_deltas

    def loss(self,
             cls_logits: torch.Tensor,
             bbox_deltas: torch.Tensor,
             proposals_list: List[torch.Tensor],
             z_boxes: List[torch.Tensor],
             x_boxes: List[torch.Tensor],
             flags: torch.Tensor, cfg: Dict, **kwargs) -> Dict[str, torch.Tensor]:
        """ Calculate the loss."""
        gt_boxes_list = generate_gt(z_boxes, x_boxes, flags, same_category_as_positive=False)

        cls_reg_targets = bbox_target(proposals_list, gt_boxes_list, self.target_means, self.target_stds, cfg.compare)
        (labels, label_weights, reg_targets, reg_weights, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg

        cls_loss = self.cls_loss_obj(cls_logits, labels, label_weights, average_factor=num_total_samples)
        reg_loss = self.reg_loss_obj(bbox_deltas, reg_targets, reg_weights, average_factor=num_total_pos)

        return dict(cls_loss=cls_loss, reg_loss=reg_loss)

    def get_boxes(self,
                  cls_logits: torch.Tensor,
                  bbox_deltas: torch.Tensor,
                  proposals: torch.Tensor,
                  cfg: Dict, **kwargs) -> torch.Tensor:
        """ Predict the bounding boxes from the network's output
        Args:
            cls_logits (torch.Tensor): in shape of [N, 2]
            bbox_deltas (torch.Tensor): in shape of [N, 4]
            proposals (torch.Tensor): in shape of [N, 5]
            cfg (dict): configuration
        """
        fm_scores = torch.softmax(cls_logits, dim=1)[:, 1]
        fm_boxes = ubox.delta2bbox(proposals[:, 0:4], bbox_deltas, means=self.target_means, stds=self.target_stds)
        dets = torch.cat((fm_boxes, fm_scores.unsqueeze(1)), dim=1)
        return dets


