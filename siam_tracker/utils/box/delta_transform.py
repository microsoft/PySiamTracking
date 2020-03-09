# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


""" Transforms between box delta & box coordinates """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np


__all__ = ['bbox2delta', 'delta2bbox', 'corner2delta', 'delta2corner']


def corner2delta(points, gt, means=(0, 0, 0, 0), stds=(1, 1, 1, 1)):
    assert len(points) == len(gt)
    dx1 = ((points[..., 0] - gt[..., 0]) - means[0]) / stds[0]
    dy1 = ((points[..., 1] - gt[..., 1]) - means[1]) / stds[1]
    dx2 = ((gt[..., 2] - points[..., 0]) - means[2]) / stds[2]
    dy2 = ((gt[..., 3] - points[..., 1]) - means[3]) / stds[3]
    deltas = torch.cat((dx1.unsqueeze(-1),
                        dy1.unsqueeze(-1),
                        dx2.unsqueeze(-1),
                        dy2.unsqueeze(-1)), dim=-1)
    return deltas


def delta2corner(points, deltas, means=(0, 0, 0, 0), stds=(1, 1, 1, 1)):
    assert len(points) == len(deltas)
    x1 = points[..., 0] - (deltas[..., 0] * stds[0] + means[0])
    y1 = points[..., 1] - (deltas[..., 1] * stds[1] + means[1])
    x2 = points[..., 0] + (deltas[..., 2] * stds[2] + means[2])
    y2 = points[..., 1] + (deltas[..., 3] * stds[3] + means[3])
    gts = torch.cat((x1.unsqueeze(-1),
                     y1.unsqueeze(-1),
                     x2.unsqueeze(-1),
                     y2.unsqueeze(-1)), dim=-1)
    return gts


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert len(proposals) == len(gt)
    is_torch = isinstance(proposals, torch.Tensor)

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    if is_torch:
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)
        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)
    else:
        dw = np.log(gw / pw)
        dh = np.log(gh / ph)
        deltas = np.stack([dx, dy, dw, dh], axis=-1)
        means = np.array(means, dtype=np.float32).reshape(1, 4)
        stds = np.array(stds, dtype=np.float32).reshape(1, 4)
        deltas = (deltas - means) / stds

    return deltas


def delta2bbox(rois, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1], wh_ratio_clip=8.0):
    assert isinstance(rois, torch.Tensor), "Support torch.Tensor only."
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes
