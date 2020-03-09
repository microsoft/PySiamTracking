# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner

from ....utils import box as ubox


class FoveaPointAssigner(BaseAssigner):
    """ FoveaPointAssigner is motivated from FoveaBox [1]. It's a point assigner.
    It's a point assigner. For each ground-truth bounding box (xc, yc, w, h), if a point (x, y) satisfy
    (1) |x - xc| < w * sigma1 * 0.5 and
    (2) |y - yc| < h * sigma1 * 0.5,
    it will be treated as positive point. Similarly, if a point (x, y) is outside the fovea region:
    (1) |x - xc| > w * sigma2 * 0.5 and
    (2) |y - yc| > h * sigma2 * 0.5,
    it will be treated as negative point. The rest of points are ignored (label -1).

    [1] FoveaBox: Beyond Anchor-based Object Detector.
    """

    def __init__(self, sigma1, sigma2):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def assign(self, boxes, gt_boxes, gt_boxes_ignore=None):
        if gt_boxes is None:
            assigned_gt_inds = boxes.new_zeros((boxes.size(0),), dtype=torch.long)
            assign_result = AssignResult(0, assigned_gt_inds, None)
            return assign_result

        # convert the format.
        cands = ubox.xyxy_to_xcycwh(boxes)
        gts = ubox.xyxy_to_xcycwh(gt_boxes)

        # [num_boxes, num_gts, 2]
        dists = torch.abs(cands[:, 0:2].view(-1, 1, 2) - gts[:, 0:2].view(1, -1, 2))

        # 1. assign -1 by default
        assigned_gt_inds = boxes.new_full((boxes.size(0),), -1, dtype=torch.long)
        min_dists, _ = dists.min(dim=1)  # [num_boxes, 2]

        _, min_dist_inds = dists.sum(dim=-1).min(dim=-1)

        pos_inds = (min_dists[:, 0] < gts[:, 2] * self.sigma1 * 0.5) & \
                   (min_dists[:, 1] < gts[:, 3] * self.sigma1 * 0.5)

        neg_inds = (min_dists[:, 0] > gts[:, 2] * self.sigma2 * 0.5) | \
                   (min_dists[:, 1] > gts[:, 3] * self.sigma2 * 0.5)

        assigned_gt_inds[pos_inds] = min_dist_inds[pos_inds] + 1
        assigned_gt_inds[neg_inds] = 0

        return AssignResult(gt_boxes.size(0), assigned_gt_inds, None)
