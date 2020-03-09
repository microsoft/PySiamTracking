# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch import Tensor
from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):

    @abstractmethod
    def assign(self, boxes: Tensor, gt_boxes: Tensor, gt_boxes_ignore=None):
        pass
