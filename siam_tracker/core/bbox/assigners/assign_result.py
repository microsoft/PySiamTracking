# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.


import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps

    def add_gt_(self):
        self_inds = torch.arange(
            1, self.num_gts + 1, dtype=torch.long, device=self.gt_inds.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
