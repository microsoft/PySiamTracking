# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Base class for some single stage trackers like SiamRPN or SiamFCOS. """

import torch

from typing import List

from .base_tracker import BaseTracker
from ..builder import build_backbone, build_head, build_fusion, build_neck
from ...utils import no_grad
from ...utils import box as ubox
from ...core import add_window_prior_to_anchors


class SingleStage(BaseTracker):

    def __init__(self,
                 backbone,
                 fusion,
                 head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SingleStage, self).__init__(train_cfg, test_cfg)
        # build backbone
        self.z_net = build_backbone(backbone)
        self.neck = None
        if neck is not None:
            self.neck = build_neck(neck)
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
        # extract convolutional features from backbone network
        z_feats = self.z_net(z_imgs)
        x_feats = self.z_net(x_imgs)
        # fuse the convolutional features
        fused_feats = self.fusion(z_feats=z_feats, x_feats=x_feats, cfg=self.train_cfg)
        network_outputs = self.head(fused_feats)
        loss_inputs = network_outputs + (z_boxes, x_boxes, flags, self.train_cfg)
        losses = self.head.loss(*loss_inputs)
        return losses

    @no_grad
    def initialize(self, img: torch.Tensor, gt_box: torch.Tensor):
        """ Extract template features from image """
        # search region image
        sr_imgs, crop_boxes = self.generate_search_region(img,
                                                          gt_box,
                                                          z_size=self.test_cfg.z_size,
                                                          x_size=self.test_cfg.x_size,
                                                          num_scales=1)
        z_feats = self.z_net(sr_imgs)
        if self.neck is not None:
            z_feats = self.neck(z_feats)
        z_boxes = self.proj_img2sr(ubox.xcycwh_to_xyxy(gt_box), crop_boxes, sr_size=self.test_cfg.x_size)
        z_boxes = z_boxes.view(1, 4).type_as(img)
        # save template information in fusion module.
        self.fusion(z_feats=z_feats, x_feats=None, z_info=z_boxes, x_info=None, cfg=self.test_cfg)
        self.tracking_results = [gt_box]

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
        x_feats = self.z_net(sr_imgs)
        if self.neck is not None:
            x_feats = self.neck(x_feats)
        fused_feats = self.fusion(z_feats=None, x_feats=x_feats, cfg=self.test_cfg)
        # apply for detection head.
        network_out = self.head(fused_feats)
        # cls_logits, bbox_deltas = self.head(fused_feats)
        head_input = network_out + (self.test_cfg, )
        # get det anchor boxes from RPN head
        dets = self.head.get_boxes(*head_input).cpu()
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


def _change(r):
    return torch.max(r, 1./r)


def _sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return torch.sqrt(sz2)
