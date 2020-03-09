# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from typing import List

from .base_tracker import BaseTracker
from ..builder import TRACKERS, build_backbone, build_fusion, build_head
from ...utils import no_grad
from ...utils import box as ubox
from ...core import add_window_prior_to_boxes
from ...ops.nms import nms


@TRACKERS.register_module
class SPM(BaseTracker):

    def __init__(self,
                 backbone,
                 cm_fusion,
                 cm_head,
                 fm_fusion,
                 fm_head,
                 train_cfg=None,
                 test_cfg=None):
        super(SPM, self).__init__(train_cfg, test_cfg)
        self.z_net = build_backbone(backbone)
        self.cm_fusion = build_fusion(cm_fusion)
        self.cm_head = build_head(cm_head)
        self.fm_fusion = build_fusion(fm_fusion)
        self.fm_head = build_head(fm_head)

    def forward(self,
                z_imgs: torch.Tensor,
                x_imgs: torch.Tensor,
                z_boxes: List[torch.Tensor],
                x_boxes: List[torch.Tensor],
                flags: List[torch.Tensor]):
        losses = dict()
        z_feats = self.z_net(z_imgs)
        x_feats = self.z_net(x_imgs)

        # CM module
        cm_fused_feats = self.cm_fusion(z_feats, x_feats, None, None, self.train_cfg)
        cm_outputs = self.cm_head(cm_fused_feats)
        cm_loss_inputs = cm_outputs + (z_boxes, x_boxes, flags, self.train_cfg)
        cm_losses = self.cm_head.loss(*cm_loss_inputs)
        for k, v in cm_losses.items():
            losses['cm_{}'.format(k)] = v

        # get boxes
        cm_boxes = self.cm_head.get_boxes(*cm_outputs, cfg=self.train_cfg)
        proposals = self.generate_proposal(cm_boxes, **self.train_cfg.proposal)
        temp_boxes = [z_box[0:4].unsqueeze(0) for z_box in z_boxes]  # [(1, 4), ...]
        # add ground-truth as proposal
        proposals = [torch.cat((x_box[0:1, 0:4], prop[:, 0:4]), dim=0) for x_box, prop in zip(x_boxes, proposals)]

        # FM module
        fm_fused_feats = self.fm_fusion(z_feats, x_feats, temp_boxes, proposals, self.train_cfg)
        fm_outputs = self.fm_head(fm_fused_feats)
        fm_loss_inputs = fm_outputs + (proposals, z_boxes, x_boxes, flags, self.train_cfg)
        fm_losses = self.fm_head.loss(*fm_loss_inputs)
        for k, v in fm_losses.items():
            losses['fm_{}'.format(k)] = v
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
        z_boxes = self.proj_img2sr(ubox.xcycwh_to_xyxy(gt_box), crop_boxes, sr_size=self.test_cfg.x_size)
        z_boxes = z_boxes.view(1, 4).type_as(img)
        # save template information in fusion module.
        self.cm_fusion(z_feats=z_feats, x_feats=None, z_info=z_boxes, cfg=self.test_cfg)
        self.fm_fusion(z_feats=z_feats, x_feats=None, z_info=z_boxes, cfg=self.test_cfg)
        self.tracking_results = [gt_box]
        return gt_box

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
        cm_fused_feats = self.cm_fusion(z_feats=None, x_feats=x_feats, z_info=None, cfg=self.test_cfg)
        # apply for detection head.
        cm_network_out = self.cm_head(cm_fused_feats)
        # cls_logits, bbox_deltas = self.head(fused_feats)
        cm_head_input = cm_network_out + (self.test_cfg, )
        # get det anchor boxes from RPN head
        cm_boxes = self.cm_head.get_boxes(*cm_head_input).view(-1, 5)
        proposals = self.generate_proposal(cm_boxes, **self.test_cfg.proposal)

        # add ground-truth as proposal
        rois = proposals[:, 0:4].contiguous()
        fm_fused_feats = self.fm_fusion(z_feats=None, x_feats=x_feats, z_info=None, x_info=rois, cfg=self.test_cfg)
        fm_cls_logits, fm_bbox_deltas = self.fm_head(fm_fused_feats)
        dets = self.fm_head.get_boxes(fm_cls_logits, fm_bbox_deltas, proposals, self.test_cfg).cpu()

        # merge the score from CM & FM
        cm_scores = proposals[:, 4].cpu()
        dets[:, 4] = dets[:, 4] * self.test_cfg.fm_score_weight + cm_scores * (1.0 - self.test_cfg.fm_score_weight)

        boxes_sr = dets[..., 0:4].contiguous()
        scores = dets[..., 4].contiguous().view(-1)

        # project to original image space
        boxes = self.proj_sr2img(boxes_sr, crop_boxes, self.test_cfg.x_size)
        boxes = ubox.xyxy_to_xcycwh(boxes.view(-1, 4))
        boxes_sr = ubox.xyxy_to_xcycwh(boxes_sr.view(-1, 4))
        # calculate the penalty size penalty and shape penalty
        s_c = _change(_sz(boxes[:, 3], boxes[:, 2]) / _sz(last_box[3], last_box[2]))
        # ratio penalty
        r_c = _change((last_box[3] / last_box[2]) / (boxes[:, 3] / boxes[:, 2]))
        penalty = torch.exp(-(s_c * r_c - 1.0) * self.test_cfg.penalty_k)
        pscores = penalty * scores

        # add cosine window score
        final_scores = add_window_prior_to_boxes(pscores, boxes_sr, self.test_cfg.x_size, **self.test_cfg.window)

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

    @no_grad
    def generate_proposal(self,
                          boxes: torch.Tensor,
                          score_thr: float = 0.0,
                          nms_iou_thr: float = 0.75,
                          max_num_per_img: int = 8) -> torch.Tensor:
        """ Generate proposals from a set of boxes.
        Args:
            boxes (torch.Tensor): in shape of [N, M, 4] or [M, 4]
            score_thr (float): score threshold
            nms_iou_thr (float): NMS Iou threshold
            max_num_per_img (int): maximum number of kept boxes.
        """
        is_single = False
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
            is_single = True
        num_imgs = boxes.size(0)
        proposals = []
        for i in range(num_imgs):
            # filter boxes
            i_boxes = boxes[i]
            if score_thr > 0:
                inds = torch.nonzero(torch.ge(i_boxes[:, -1], score_thr)).view(-1)
                i_boxes = i_boxes[inds].contiguous()
            # apply for NMS
            i_boxes = nms(i_boxes, nms_iou_thr)
            if len(i_boxes) > max_num_per_img:
                i_boxes = i_boxes[:max_num_per_img]
            proposals.append(i_boxes)
        if is_single:
            proposals = proposals[0]
        return proposals


def _change(r):
    return torch.max(r, 1./r)


def _sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return torch.sqrt(sz2)
