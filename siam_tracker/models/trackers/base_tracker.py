# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch import nn

from typing import Tuple

from ...utils.crop import roi_crop
from ...utils.box import xcycwh_to_xywh, xcycwh_to_xyxy


class BaseTracker(nn.Module):
    """ Base class for SiameseNetwork based tracking. """
    def __init__(self, train_cfg=None, test_cfg=None):
        super(BaseTracker, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.tracking_results = []

    def freeze_block(self, num_blocks: int):
        """ Freeze the backbone network during tracking. """
        for m in self.modules():
            if id(self) == id(m):
                continue
            if hasattr(m, 'freeze_block'):
                m.freeze_block(num_blocks)

    def set_phase(self, is_training: bool = False):
        """ Setup tracker status: for training or for inference. """
        if not is_training:
            self.eval()
        else:
            self.train()
            if hasattr(self.train_cfg, 'num_freeze_blocks') and self.train_cfg.num_freeze_blocks > 0:
                self.freeze_block(self.train_cfg.num_freeze_blocks)

    def initialize(self, img: torch.Tensor, gt_box: torch.Tensor) -> None:
        """ Core function for a tracker.
        This function will be called firstly when start tracking a new sequence.
        Args:
            img (torch.Tensor): image tensor (normalized) in shape of [1, 3, H, W]
            gt_box (torch.Tensor): [xc, yc, w, h], in shape of [1, 4]
        """
        raise NotImplementedError

    def predict(self, img: torch.Tensor, gt_box: torch.Tensor = None) -> torch.Tensor:
        """ Core function for a tracker.
        This function is called in every frame.
        Args:
            img (torch.Tensor): in shape of [1, 3, H, W]
            gt_box (torch.Tensor): [xc, yc, w, h], in shape of [1, 4]
        Returns:
            track_box (torch.Tensor): [xc, yc, w, h] in shape of [1, 4]
        """
        raise NotImplementedError

    def generate_search_region(self,
                               img: torch.Tensor,
                               box: torch.Tensor,
                               z_size: int = 127,
                               x_size: int = 255,
                               context_amount: float = 0.5,
                               num_scales: int = 1,
                               scale_step: float = 1.0375) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Genearet a search region image according to the box coordinates.

        This function is widely used in Siamese-network based trackers. When inference, 'box' is usually
        the tracking result in the last frame.

        Args:
            img (torch.Tensor): the input image from which to crop. [1, 3, h, w]
            box (torch.Tensor): the target coordinates in [xc, yc, w, h]
            z_size (int): template image size (typically 127)
            x_size (int): search region image size (typically around 255)
            context_amount (float): context ratio
            num_scales (int): number of search region scales.
            scale_step (float): scale step.

        Returns:
            crop_imgs (torch.Tensor): the search region image. [num_scales, 3, x_size, x_size]
            crop_boxes (torch.Tensor): search region boxes. [num_scales, 4] (x1, y1, w, h)
        """
        if box.dim() == 2:
            assert box.size(0) == 1
            box = box.squeeze(0)
        scale_factors = scale_step ** (torch.arange(num_scales, dtype=torch.float32) - (num_scales - 1.) / 2.)

        base_z_context_size = box[2:4] + context_amount * box[2:4].sum()
        base_z_size = base_z_context_size.prod().sqrt()
        base_x_size = (x_size / z_size) * base_z_size

        crop_boxes_size = scale_factors.view(-1, 1) * base_x_size.view(1, -1)
        crop_boxes_ctr = box[0:2].view(1, 2).expand(num_scales, 2)
        crop_boxes = torch.cat([crop_boxes_ctr, crop_boxes_size, crop_boxes_size], dim=1)  # [nscales, 4]

        rois = xcycwh_to_xyxy(crop_boxes)
        crop_imgs = roi_crop(img, rois, self.test_cfg.x_size, self.test_cfg.x_size)
        crop_boxes = xcycwh_to_xywh(crop_boxes)
        # convert to xywh, because it's more convenient to calculate the mapping from search region to image.
        return crop_imgs, crop_boxes

    @staticmethod
    def proj_img2sr(boxes: torch.Tensor,
                    crop_boxes: torch.Tensor,
                    sr_size: int) -> torch.Tensor:
        """ Project bounding-box from img space to search region space.

        Args:
            boxes (torch.Tensor): [num_box, 4], in order of (x1, y1, x2, y2)
            crop_boxes (torch.Tensor): [num_scale, 4], (x1, y1, w, h)
            sr_size (int): search region size.
        Returns:
            boxes_in_sr (torch.Tensor): [num_scale, num_box, 4]
        """
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
        x1y1 = crop_boxes[:, 0:2].repeat(1, 2).unsqueeze(1)  # [n_scales, 1, 4]
        wh = crop_boxes[:, 2:4].repeat(1, 2).unsqueeze(1)  # [n_scales, 1, 4]
        boxes = (boxes.unsqueeze(0) - x1y1) * sr_size / wh
        return boxes

    @staticmethod
    def proj_sr2img(boxes: torch.Tensor,
                    crop_boxes: torch.Tensor,
                    sr_size: int) -> torch.Tensor:
        """ Project bbox coordinate from search region space to image space.
        Args:
            boxes (torch.Tensor): [num_scale, num_boxes, 4] (x1, y1, x2, y2)
            crop_boxes (torch.Tensor): [num_scale, 4] (x1, y1, w, h)
            sr_size (int): search region size
        Args:
            boxes (torch.Tensor): [num_scale, num_boxes, 4] (x1, y1, x2, y2)
        """
        if boxes.dim() == 2:
            assert crop_boxes.size(0) == 1
            x1y1 = crop_boxes[:, 0:2].repeat(1, 2)  # 1, 4
            wh = crop_boxes[:, 2:4].repeat(1, 2)  # [1, 4]
        elif crop_boxes.dim() == 2 and boxes.dim() == 3:
            x1y1 = crop_boxes[:, 0:2].repeat(1, 2).unsqueeze(1)  # [n_scales, 1, 4]
            wh = crop_boxes[:, 2:4].repeat(1, 2).unsqueeze(1)  # [n_scales, 1, 4]
        else:
            raise ValueError("unknown values. {}".format(crop_boxes.size()))
        boxes = (boxes * wh / sr_size) + x1y1
        return boxes
