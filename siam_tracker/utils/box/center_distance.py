# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import torch

from .coordinate import xyxy_to_xcycwh


def bbox_center_dist(bboxes1, bboxes2, mode='L1'):
    """Calculate center distance between two set of bboxes.

        Args:
            bboxes1 (Tensor): shape (m, 4)
            bboxes2 (Tensor): shape (n, 4)

        Returns:
            dist(Tensor): shape (m, n)
    """

    assert mode in ['L1', 'L2']
    ctr1 = xyxy_to_xcycwh(bboxes1)[:, 0:2] # [m, 2]
    ctr2 = xyxy_to_xcycwh(bboxes2)[:, 0:2] # [n, 2]

    delta = ctr1.unsqueeze(1) - ctr2.unsqueeze(0)  # [m, n, 2]
    if mode == 'L1':
        dist = torch.abs(delta).sum(dim=-1)  # [m, n]
    elif mode == 'L2':
        dist = torch.sqrt((delta ** 2).sum(dim=-1))
    else:
        raise NotImplementedError
    return dist
