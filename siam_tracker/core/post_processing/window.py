# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from typing import Tuple


def get_window_prior_from_score_maps(score_maps):
    height, width = score_maps.size()[-2:]
    _win_h = torch.hann_window(height, periodic=False, dtype=torch.float32)
    _win_w = torch.hann_window(width, periodic=False, dtype=torch.float32)
    win_2d = torch.matmul(_win_h.unsqueeze(-1), _win_w.unsqueeze(0)).type_as(score_maps)
    if score_maps.dim() == 3:
        win_2d = win_2d.view(1, height, width)
    elif score_maps.dim() == 4:
        win_2d = win_2d.view(1, 1, height, width)
    return win_2d


def add_window_prior_to_score_maps(score_maps, weight=0.5):
    win_2d = get_window_prior_from_score_maps(score_maps)
    score_maps = win_2d * weight + score_maps * (1 - weight)
    return score_maps


def add_window_prior_to_anchors(scores: torch.Tensor,
                                featmap_size: Tuple,
                                weight: float = 0.5) -> torch.Tensor:
    feat_height, feat_width = featmap_size
    _win_h = torch.hann_window(feat_height, periodic=False, dtype=torch.float32)
    _win_w = torch.hann_window(feat_width, periodic=False, dtype=torch.float32)
    win_2d = torch.matmul(_win_h.unsqueeze(-1), _win_w.unsqueeze(0)).type_as(scores)

    num_repeats = int(scores.size(0) / (feat_height * feat_width))
    win_2d = win_2d.view(feat_height, feat_width, 1).repeat(1, 1, num_repeats).view(-1)
    scores = scores * (1 - weight) + win_2d * weight

    return scores


def add_window_prior_to_boxes(scores: torch.Tensor,
                              boxes: torch.Tensor,
                              x_size: int,
                              norm_size: int,
                              weight: float = 0.5) -> torch.Tensor:
    """ Add cosein windows penalty to a series of bounding boxes.
    Args:
        scores (torch.Tensor): in shape of [N, ]
        boxes (torch.Tensor): in shape of [N, 4], the order is [xc, yc, w, h]
        x_size (int): search region size.
        norm_size (int): normalization size
        weight (float): cosine windows weight.
    """
    center_dist = (boxes[:, 0:2] - (x_size - 1) / 2) / (norm_size - 1) + 0.5
    center_dist.clamp_(min=0.0, max=1.0)
    win_scores_2d = 0.5 - torch.cos(center_dist * 2.0 * 3.1415) * 0.5
    win_scores = win_scores_2d[:, 0] * win_scores_2d[:, 1]
    scores = scores * (1 - weight) + win_scores * weight
    return scores
