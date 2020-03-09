# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch import nn, Tensor
from typing import Dict, Any, List

from .base_fusion import BaseFusion
from .xcorr import CrossCorrelation
from ..builder import FUSIONS, build_fusion


@FUSIONS.register_module
class MultiLevelFusion(BaseFusion):

    def __init__(self,
                 fusions: List):
        super(MultiLevelFusion, self).__init__()
        self.fusions = nn.ModuleList(
            [build_fusion(fusions[i]) for i in range(len(fusions))]
        )

    def extract_z_feat(self,
                       z_feats: Dict[str, Tensor],
                       z_info: Any,
                       cfg: Dict):
        z_feat_list = []
        for op in self.fusions:
            i_z_feat = op.extract_z_feat(z_feats, z_info, cfg)
            z_feat_list.append(i_z_feat)
        return z_feat_list

    def extract_x_feat(self,
                       x_feats: Dict[str, Tensor],
                       x_info: Any,
                       cfg: Dict):
        x_feat_list = []
        for op in self.fusions:
            i_x_feat = op.extract_x_feat(x_feats, x_info, cfg)
            x_feat_list.append(i_x_feat)
        return x_feat_list

    def fuse(self,
             z_feat,
             x_feat):
        return [op.fuse(z, x) for z, x, op in zip(z_feat, x_feat, self.fusions)]
