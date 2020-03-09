# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch import nn, Tensor

from typing import Dict, Union, List, Any


class BaseFusion(nn.Module):
    """ Base fusion module for SiameseNetwork.
    In this implementation, the entire siamese network can be divided into three
    parts: backbone, fusion and head. The organization is shown as following:

    z_img -- backbone ---
                         \
                          Fusion ---> Head
                         /
    x_img -- backbone ---

    """
    def __init__(self):
        super(BaseFusion, self).__init__()
        self.z_cache = None
        self.z_init_cache = None

    def forward(self,
                z_feats: Dict[str, Tensor] = None,
                x_feats: Dict[str, Tensor] = None,
                z_info: Any = None,
                x_info: Any = None,
                cfg: Dict = None) -> Tensor:
        """ Fuse template features and search region features.
        The inputs have 3 situations:

        1) z_feats != None and x_feats == None: extract template features and save them.
        2) z_feats == None and x_feats != None: fuse the search region feats and cached temp feats.
        3) z_feats != None and x_feats != None: fuse the search region feats and temp feats.

        Args:
            z_feats (Dict): template feature dict. extracted from backbone.
            x_feats (Dict): search region feature dict. extracted from backbone.
            z_info (Any): some necessary information to extract template feature
            x_info (Any): some necessary information to extract search region feature.
            cfg (Dict): training or testing configuration.
        """
        if z_feats is None and x_feats is not None:
            # load template features from cache
            x_feat = self.extract_x_feat(x_feats, x_info, cfg)
            return self.fuse(self.z_cache, x_feat)
        elif x_feats is None and z_feats is not None:
            self.z_cache = self.extract_z_feat(z_feats, z_info, cfg)
            self.z_init_cache = self.z_cache
        elif z_feats is not None and x_feats is not None:
            z_feat = self.extract_z_feat(z_feats, z_info, cfg)
            x_feat = self.extract_x_feat(x_feats, x_info, cfg)
            return self.fuse(z_feat, x_feat)
        else:
            raise ValueError("At least one element of z_feats and x_feats should be NOT NONE.")

    def extract_z_feat(self,
                       z_feats: Dict[str, Tensor],
                       z_info: Any,
                       cfg: Dict) -> Union[Tensor, List]:
        raise NotImplementedError

    def extract_x_feat(self,
                       x_feats: Dict[str, Tensor],
                       x_info: Any,
                       cfg: Dict) -> Union[Tensor, List]:
        raise NotImplementedError

    def fuse(self,
             z_feat: Tensor,
             x_feat: Tensor) -> Union[Tensor, List]:
        raise NotImplementedError

    def linear_update(self,
                      z_feat: Union[Tensor, List],
                      init_portion: float = 0.5,
                      gamma: float = 0.975):

        def update_single(new, cached, init, p, g):
            if isinstance(new, Tensor):
                return init * p + (1 - p) * g * cached + (1 - p) * (1 - g) * new
            else:
                return [update_single(n, c, i, p, g) for n, c, i in zip(new, cached, init)]

        self.z_cache = update_single(z_feat, self.z_cache, self.z_init_cache, init_portion, gamma)


