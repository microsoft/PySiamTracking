# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner

from ....utils import box as ubox


class CenterDistAssigner(BaseAssigner):
    """ Assign label according to the center point distances.
    In this way, the box width & height are ignored. This strategy is mainly used in SiamFC tracker.
    """

    def __init__(self, pos_thresh, neg_thresh, dist_type='L1'):
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.dist_type = dist_type

    def assign(self, boxes, gt_boxes, gt_boxes_ignore=None):
        if gt_boxes is None:
            assigned_gt_inds = boxes.new_zeros((boxes.size(0),), dtype=torch.long)
            assign_result = AssignResult(0, assigned_gt_inds, None)
            return assign_result

        dists = ubox.bbox_center_dist(boxes, gt_boxes, mode=self.dist_type)  # [N, 1]
        min_dists, _ = dists.min(dim=1)

        # 1. assign -1 by default
        assigned_gt_inds = boxes.new_full((boxes.size(0),), -1, dtype=torch.long)

        pos_inds = min_dists < self.pos_thresh
        assigned_gt_inds[pos_inds] = 1
        neg_inds = min_dists > self.neg_thresh
        assigned_gt_inds[neg_inds] = 0

        return AssignResult(gt_boxes.size(0), assigned_gt_inds, min_dists)
