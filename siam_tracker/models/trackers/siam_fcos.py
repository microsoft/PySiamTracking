# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .single_stage import SingleStage
from ..builder import TRACKERS


@TRACKERS.register_module
class SiamFCOS(SingleStage):

    def __init__(self,
                 backbone,
                 fusion,
                 head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SiamFCOS, self).__init__(backbone, fusion, head, neck, train_cfg, test_cfg)
