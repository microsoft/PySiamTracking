# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.


import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        if gt_bboxes is not None:
            self.num_gts = gt_bboxes.shape[0]
            self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        else:
            self.num_gts = 0
            self.pos_assigned_gt_inds = torch.zeros(0, dtype=torch.long, device=pos_inds.device)
            self.pos_gt_bboxes = torch.zeros(0, 4, dtype=torch.float32, device=pos_inds.device)

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
