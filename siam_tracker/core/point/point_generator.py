# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from ...utils.box import xyxy_to_xcycwh, xcycwh_to_xyxy


class PointGenerator(object):

    @staticmethod
    def grid_points(featmap_size, stride=16, device='cpu'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_x = shift_x - (feat_w - 1) / 2.0 * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_y = shift_y - (feat_h - 1) / 2.0 * stride
        shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)
        points = torch.stack([shift_xx, shift_yy], dim=-1).view(-1, 2)
        return points

    @staticmethod
    def assign_prior_boxes(points, boxes):
        # convert to boxes from xyxy to xcycwh
        boxes = xyxy_to_xcycwh(boxes)
        wh = boxes[:, 2:4]
        num_boxes = wh.size(0)
        num_pts = points.size(0)
        wh = wh.view(1, num_boxes, 2).expand(num_pts, num_boxes, 2)
        points = points.view(num_pts, 1, 2).expand(num_pts, num_boxes, 2)
        final = torch.cat([points, wh], dim=-1).view(-1, 4)
        final = xcycwh_to_xyxy(final)
        return final
