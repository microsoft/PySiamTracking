import torch
from typing import List, Dict

from .assign_sampling import assign_and_sample

from ...utils import multi_apply
from ...utils import box as ubox


def bbox_target(boxes_list: List[torch.Tensor],
                gt_boxes_list: List[torch.Tensor],
                target_means: List = (0., 0., 0., 0.),
                target_stds: List = (1., 1., 1., 1.),
                cfg: Dict = None):
    """ Assign the category label to each boxes and calculate the regression target.
    Args:
        boxes_list (List[torch.Tensor]): candidate boxes list. each element is a [K, 4] tensor.
        gt_boxes_list (List[torch.Tensor]): ground-truth boxes list. each element is a [M, 4] tensor.
        target_means (List): regression means.
        target_stds (List): regression stds
        cfg (dict): training configurations
    """
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     pos_inds_list, neg_inds_list) = multi_apply(
        bbox_target_single,
        boxes_list,
        gt_boxes_list,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg)
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

    # group into one
    labels = torch.cat([_ for _ in labels_list], dim=0)
    label_weights = torch.cat([_ for _ in label_weights_list], dim=0)
    bbox_targets = torch.cat([_ for _ in bbox_targets_list], dim=0)
    bbox_weights = torch.cat([_ for _ in bbox_weights_list], dim=0)

    return labels, label_weights, bbox_targets, bbox_weights, num_total_pos, num_total_neg


def bbox_target_single(boxes: torch.Tensor,
                       gt_boxes: torch.Tensor,
                       target_means: List,
                       target_stds: List,
                       cfg: Dict):
    assign_result, sampling_result = assign_and_sample(boxes, gt_boxes, None, cfg)
    num_valid_boxes = boxes.size(0)
    bbox_targets = torch.zeros_like(boxes)
    bbox_weights = torch.zeros_like(boxes)
    labels = boxes.new_zeros(num_valid_boxes, dtype=torch.long)
    label_weights = boxes.new_zeros(num_valid_boxes, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    if len(pos_inds) > 0:
        pos_bbox_targets = ubox.bbox2delta(sampling_result.pos_bboxes,
                                           sampling_result.pos_gt_bboxes,
                                           target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        if 'bbox_inner_weight' in cfg:
            pw = cfg.bbox_inner_weight
        else:
            pw = 1.0
        bbox_weights[pos_inds, :] = pw
        labels[pos_inds] = 1
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

