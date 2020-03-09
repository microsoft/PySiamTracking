# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch.nn import functional as F

from ..builder import LOSSES


@LOSSES.register_module
class BinaryCrossEntropyLoss(object):

    def __init__(self,
                 loss_weight=1.0):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                average_factor=None):
        target.clamp_(min=0.0, max=1.0)
        n = pred.size(0)
        all_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        if weight is not None:
            all_loss = all_loss * weight
        if average_factor is None:
            assert weight is not None
            average_factor = weight.sum().clamp(0.01).item()
        mean_loss = all_loss.sum() / average_factor
        return mean_loss * self.loss_weight

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@LOSSES.register_module
class CrossEntropyLoss(object):

    def __init__(self,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                average_factor=None):
        target.clamp_(min=0)
        raw = F.cross_entropy(pred, target, reduction='none')

        n = pred.size(0)
        if weight is not None:
            raw = raw * weight
        if average_factor is None:
            assert weight is not None
            average_factor = weight.sum().clamp(0.01).item()
        mean_loss = raw.sum() / average_factor
        return mean_loss * self.loss_weight

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
