# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" SiamFC Tracker
Fully-Convolutional Siamese Networks for Object Tracking, ECCVW 2016.
https://www.robots.ox.ac.uk/~luca/siamese-fc.html
"""

import torch
from torch.nn.functional import interpolate

from typing import List

from .base_tracker import BaseTracker
from ..builder import TRACKERS, build_backbone, build_head, build_fusion, build_neck
from ...core import get_window_prior_from_score_maps
from ...utils import no_grad
from ...utils import box as ubox


@TRACKERS.register_module
class SiamFC(BaseTracker):

    def __init__(self,
                 backbone,
                 fusion,
                 head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SiamFC, self).__init__(train_cfg, test_cfg)
        # build backbone
        self.z_net = build_backbone(backbone)
        # build neck if necessary
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        # build fusion module
        self.fusion = build_fusion(fusion)
        # build head
        self.head = build_head(head)

    def forward(self,
                z_imgs: torch.Tensor,
                x_imgs: torch.Tensor,
                z_boxes: List[torch.Tensor],
                x_boxes: List[torch.Tensor],
                flags: List[torch.Tensor]):
        """ This function is called during training."""
        assert z_imgs.dim() == 4 and x_imgs.dim() == 4  # [N, 3, H, W]
        # extract convolutional features from backbone network
        z_feats = self.z_net(z_imgs)
        x_feats = self.z_net(x_imgs)
        if self.neck is not None:
            z_feats = self.neck(z_feats)
            x_feats = self.neck(x_feats)
        # fuse the convolutional features
        fused_feats = self.fusion(z_feats, x_feats, None, None, self.train_cfg)
        # apply for SiamFC head.
        cls_logits = self.head(fused_feats)
        loss = self.head.loss(cls_logits, z_boxes, x_boxes, flags, self.train_cfg)
        return loss

    @no_grad
    def initialize(self, img, box):
        """ Extract template features from image """
        # search region image
        sr_imgs, crop_boxes = self.generate_search_region(img,
                                                          box,
                                                          z_size=self.test_cfg.z_size,
                                                          x_size=self.test_cfg.x_size,
                                                          num_scales=1)
        z_feats = self.z_net(sr_imgs)
        z_boxes = self.proj_img2sr(ubox.xcycwh_to_xyxy(box), crop_boxes, sr_size=self.test_cfg.x_size)
        z_boxes = z_boxes.view(1, 4).type_as(img)
        self.fusion(z_feats=z_feats, x_feats=None, z_info=z_boxes, cfg=self.test_cfg)
        self.tracking_results = [box]

    @no_grad
    def predict(self, img, gt_box=None):
        assert isinstance(self.tracking_results, list) and len(self.tracking_results) >= 1
        last_box = self.tracking_results[-1].view(4)
        # generate search region image
        sr_imgs, crop_boxes = self.generate_search_region(img,
                                                          last_box,
                                                          z_size=self.test_cfg.z_size,
                                                          x_size=self.test_cfg.x_size,
                                                          **self.test_cfg.search_region)
        x_feats = self.z_net(sr_imgs)
        fused_feats = self.fusion(z_feats=None, x_feats=x_feats, z_info=None, cfg=self.test_cfg)
        response_maps = self.head(fused_feats)
        # upsample response map so that we can get a finer location prediction
        response_maps = interpolate(response_maps,
                                    **self.test_cfg.upsampler).squeeze(1)  # [3, H, W]
        response_maps = response_maps.cpu()  # move to cpu

        num_scales = response_maps.size(0)
        if num_scales > 1:
            # select the best scale
            penalty = torch.arange(num_scales, dtype=torch.float) - (num_scales - 1.) / 2.
            penalty = self.test_cfg.scale_penalty ** torch.abs(penalty)
            max_scores = penalty * (response_maps.view(response_maps.size(0), -1).max(dim=1)[0])
            _, best_scale_idx = torch.max(max_scores, dim=0)
            best_scale_idx = int(best_scale_idx)
        else:
            best_scale_idx = 0
        response_map = response_maps[best_scale_idx]
        crop_box = crop_boxes[best_scale_idx]

        # response_map = (response_map - response_map.min()) / response_map.sum()
        # normalize the scores into [0, 1]
        response_map = response_map - response_map.min()
        r_sum = response_map.sum()
        if r_sum > 0:
            response_map = response_map / r_sum
        # add cosine windows
        win_prior = get_window_prior_from_score_maps(response_map)
        win_prior = win_prior / win_prior.sum()
        response_map = response_map * (1 - self.test_cfg.window.weight) + win_prior * self.test_cfg.window.weight

        # select a best position
        max_idx = int(response_map.view(-1).max(0)[1])
        max_y, max_x = max_idx // response_map.size(1), max_idx % response_map.size(1)
        ctr_y, ctr_x = (response_map.size(0) - 1) / 2.0, (response_map.size(1) - 1) / 2.0
        best_pos = torch.tensor([max_x - ctr_x, max_y - ctr_y], dtype=torch.float32)

        # from search region to image
        x_scale = crop_box[2:4] / self.test_cfg.x_size
        xy_delta = best_pos / self.test_cfg.upsampler.scale_factor * self.head.stride * x_scale
        xy = last_box[0:2] + xy_delta

        # calculate the scale
        scale_ctr_idx = best_scale_idx - (num_scales - 1) // 2
        if scale_ctr_idx == 0:
            # same size with last box
            wh = last_box[2:4]
        else:
            scale_ratio = self.test_cfg.search_region.scale_step ** scale_ctr_idx
            # linear interpolation so that it won't change fast
            scale_ratio = (1 - self.test_cfg.scale_damp) + self.test_cfg.scale_damp * scale_ratio
            wh = last_box[2:4] * scale_ratio
        wh = self.tracking_results[0][0, 2:4] * torch.clamp(wh / self.tracking_results[0][0, 2:4], min=0.2, max=5.0)
        result = torch.cat([xy, wh]).unsqueeze(0)
        self.tracking_results.append(result)

        if 'linear_update' in self.test_cfg and self.test_cfg.update is not None:
            if self.test_cfg.linear_update.enable:
                p = self.test_cfg.linear_update.get('init_portion', 0.5)
                gamma = self.test_cfg.linear_update.get('gamma', 0.975)
                # generate search region for current frame
                last_box = self.tracking_results[-1].view(4)
                # generate search region image
                sr_imgs, crop_boxes = self.generate_search_region(img,
                                                                  last_box,
                                                                  z_size=self.test_cfg.z_size,
                                                                  x_size=self.test_cfg.x_size,
                                                                  num_scales=1)
                z_feats = self.z_net(sr_imgs)
                z_boxes = self.proj_img2sr(ubox.xcycwh_to_xyxy(last_box), crop_boxes, sr_size=self.test_cfg.x_size)
                z_boxes = z_boxes.view(1, 4).type_as(img)
                z_roi_feat = self.fusion.extract_z_feat(z_feats, z_boxes, self.test_cfg)
                self.fusion.linear_update(z_roi_feat, init_portion=p, gamma=gamma)

        return result
