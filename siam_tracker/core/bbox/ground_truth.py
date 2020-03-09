# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch


def generate_gt(z_box, x_box, flag, same_category_as_positive=False):
    """ Generate the postive ground-truth boxes for templates
    If same_category_as_postive is False, only the object pairs with same
    instance flag will be considered as positives. Otherwise, the objects in
    the same category will be viewed as positives, too.

    Args:
        z_box (list of torch.Tensor[6]): the template boxes, [x1, y1, x2, y2, cat_id, ignore_label]
        x_box (list of torch.Tensor[N, 6]): similar to z_box,
        flag (torch.Tensor [M]): if the m-th pairs are in same instance,
        same_category_as_positive (bool): if treat same category as positive.

    Returns:
        gt_box (list of torch.Tensor[N, 4]): the positive boxes in search regions.
    """
    nbatch = len(z_box)
    gt_box = []
    for i in range(nbatch):
        if same_category_as_positive:
            z_cat_id = z_box[i][4]
            same_cat_inds = torch.nonzero(torch.eq(x_box[i][:, 4], z_cat_id)).view(-1)
            if len(same_cat_inds) > 0:
                gt_box.append(x_box[i][same_cat_inds, 0:4].contiguous())
            else:
                gt_box.append(None)
        else:
            if flag[i] < 1:
                gt_box.append(None)
            else:
                gt_box.append(x_box[i][0:1, 0:4].contiguous())
    return gt_box
