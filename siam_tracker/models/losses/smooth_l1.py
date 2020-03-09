# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch


from ..builder import LOSSES


def smooth_l1_loss(pred, target, weight, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss * weight


@LOSSES.register_module
class SmoothL1Loss(object):

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                average_factor=None,
                **kwargs):
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            **kwargs)

        if average_factor is None:
            assert weight is not None
            average_factor = torch.sum(weight > 0).float().item() + 1e-6

        return loss_bbox.sum()[None] / average_factor

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@LOSSES.register_module
class L1Loss(object):

    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                average_factor=None,
                **kwargs):
        loss_bbox = torch.abs(pred - target) * weight * self.loss_weight

        if average_factor is None:
            assert weight is not None
            average_factor = torch.sum(weight > 0).float().item() + 1e-6

        return loss_bbox.sum()[None] / average_factor

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
