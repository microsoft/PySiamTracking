# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" SA-Siam Tracker """

import torch
from torch.nn.functional import interpolate

from typing import List

from .base_tracker import BaseTracker
from ..builder import TRACKERS, build_backbone, build_head, build_fusion

from ...core import get_window_prior_from_score_maps, add_window_prior_to_anchors
from ...utils import no_grad
from ...utils import box as ubox


class SASiamBase(BaseTracker):

    def __init__(self,
                 s_backbone,
                 s_fusion,
                 s_head,
                 a_backbone,
                 a_fusion,
                 a_head,
                 train_cfg=None,
                 test_cfg=None):
        super(SASiamBase, self).__init__(train_cfg, test_cfg)
        self.s_net = build_backbone(s_backbone)
        self.s_fusion = build_fusion(s_fusion)
        self.s_head = build_head(s_head)

        self.a_net = build_backbone(a_backbone)
        self.a_fusion = build_fusion(a_fusion)
        self.a_head = build_head(a_head)

    def forward(self,
                z_imgs: torch.Tensor,
                x_imgs: torch.Tensor,
                z_boxes: List[torch.Tensor],
                x_boxes: List[torch.Tensor],
                flags: List[torch.Tensor]):
        """ This function is called during training."""
        losses = dict()

        z_feats = self.s_net(z_imgs)
        x_feats = self.s_net(x_imgs)
        fused_feats = self.s_fusion(z_feats, x_feats, None, None, self.train_cfg)
        output = self.s_head(fused_feats)
        if isinstance(output, torch.Tensor):
            loss_input = (output, z_boxes, x_boxes, flags, self.train_cfg)
        else:
            loss_input = output + (z_boxes, x_boxes, flags, self.train_cfg)
        s_net_loss = self.s_head.loss(*loss_input)
        for k, v in s_net_loss.items():
            losses['s_net_{}'.format(k)] = v

        z_feats = self.a_net(z_imgs)
        x_feats = self.a_net(x_imgs)
        fused_feats = self.a_fusion(z_feats, x_feats, None, None, self.train_cfg)
        output = self.a_head(fused_feats)
        if isinstance(output, torch.Tensor):
            loss_input = (output, z_boxes, x_boxes, flags, self.train_cfg)
        else:
            loss_input = output + (z_boxes, x_boxes, flags, self.train_cfg)
        a_net_loss = self.a_head.loss(*loss_input)
        for k, v in a_net_loss.items():
            losses['a_net_{}'.format(k)] = v

        return losses

    @no_grad
    def initialize(self, img, box):
        """ Extract template features from image """
        # search region image
        sr_imgs, crop_boxes = self.generate_search_region(img,
                                                          box,
                                                          z_size=self.test_cfg.z_size,
                                                          x_size=self.test_cfg.x_size,
                                                          num_scales=1)
        z_boxes = self.proj_img2sr(ubox.xcycwh_to_xyxy(box), crop_boxes, sr_size=self.test_cfg.x_size)
        z_boxes = z_boxes.view(1, 4).type_as(img)

        z_feats = self.s_net(sr_imgs)
        self.s_fusion(z_feats=z_feats, x_feats=None, z_info=z_boxes, cfg=self.test_cfg)

        z_feats = self.a_net(sr_imgs)
        self.a_fusion(z_feats=z_feats, x_feats=None, z_info=z_boxes, cfg=self.test_cfg)

        self.tracking_results = [box]

    def freeze_block(self, num_blocks: int):
        self.s_net.freeze_block(num_blocks)


@TRACKERS.register_module
class SASiamRPN(SASiamBase):

    def __init__(self, *args, **kwargs):
        super(SASiamRPN, self).__init__(*args, **kwargs)

    @no_grad
    def predict(self, img: torch.Tensor, gt_box: torch.Tensor = None):
        assert isinstance(self.tracking_results, list) and len(self.tracking_results) >= 1
        last_box = self.tracking_results[-1].view(4)
        # generate search region image
        sr_imgs, crop_boxes = self.generate_search_region(img,
                                                          last_box,
                                                          z_size=self.test_cfg.z_size,
                                                          x_size=self.test_cfg.x_size,
                                                          **self.test_cfg.search_region)

        x_feats = self.s_net(sr_imgs)
        fused_feats = self.s_fusion(z_feats=None, x_feats=x_feats, cfg=self.test_cfg)
        # apply for detection head.
        network_out = self.s_head(fused_feats)
        head_input = network_out + (self.test_cfg, )
        s_dets = self.s_head.get_boxes(*head_input).cpu()

        x_feats = self.a_net(sr_imgs)
        fused_feats = self.a_fusion(z_feats=None, x_feats=x_feats, cfg=self.test_cfg)
        # apply for detection head.
        network_out = self.a_head(fused_feats)
        head_input = network_out + (self.test_cfg,)
        a_dets = self.a_head.get_boxes(*head_input).cpu()

        dets = (s_dets + a_dets) * 0.5
        boxes = dets[..., 0:4].contiguous()
        scores = dets[..., 4].contiguous().view(-1)

        # project to original image space
        boxes = self.proj_sr2img(boxes, crop_boxes, self.test_cfg.x_size)
        boxes = ubox.xyxy_to_xcycwh(boxes.view(-1, 4))

        # calculate the penalty size penalty and shape penalty
        s_c = _change(_sz(boxes[:, 3], boxes[:, 2]) / _sz(last_box[3], last_box[2]))
        # ratio penalty
        r_c = _change((last_box[3] / last_box[2]) / (boxes[:, 3] / boxes[:, 2]))
        penalty = torch.exp(-(s_c * r_c - 1.0) * self.test_cfg.penalty_k)
        pscores = penalty * scores

        # add cosine window score
        final_scores = add_window_prior_to_anchors(pscores,
                                                   network_out[0].size()[2:4],
                                                   **self.test_cfg.window)

        # select the best candidate
        best_id = torch.argmax(final_scores)
        select_box = boxes[best_id]
        if pscores[best_id] < self.test_cfg.min_score_threshold:
            select_box = last_box.clone()
        else:
            lr = self.test_cfg.linear_inter_rate
            select_box[2:4] = select_box[2:4] * lr + last_box[2:4] * (1 - lr)
            select_box[0].clamp_(min=0, max=img.size(3))  # x center
            select_box[1].clamp_(min=0, max=img.size(2))  # y center
            select_box[2].clamp_(min=self.test_cfg.min_box_size, max=img.size(3))  # w
            select_box[3].clamp_(min=self.test_cfg.min_box_size, max=img.size(2))  # h

        self.tracking_results.append(select_box.unsqueeze(0))

        return select_box


@TRACKERS.register_module
class SASiamFC(SASiamBase):

    def __init__(self, *args, **kwargs):
        super(SASiamFC, self).__init__(*args, **kwargs)

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
        x_feats = self.s_net(sr_imgs)
        fused_feats = self.s_fusion(z_feats=None, x_feats=x_feats, cfg=self.test_cfg)
        s_response_maps = self.s_head(fused_feats)

        x_feats = self.a_net(sr_imgs)
        fused_feats = self.a_fusion(z_feats=None, x_feats=x_feats, cfg=self.test_cfg)
        a_response_maps = self.a_head(fused_feats)

        response_maps = (s_response_maps + a_response_maps) * 0.5
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
        xy_delta = best_pos / self.test_cfg.upsampler.scale_factor * self.a_head.stride * x_scale
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
        return result


def _change(r):
    return torch.max(r, 1. / r)


def _sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return torch.sqrt(sz2)
