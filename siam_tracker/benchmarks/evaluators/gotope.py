import numpy as np

from .ope import OPE
from ..builder import EVALUATORS


@EVALUATORS.register_module
class GOTOPE(OPE):

    def __init__(self, *args, **kwargs):
        super(GOTOPE, self).__init__(*args, **kwargs)

    def evaluate(self, info, *args, **kwargs):
        results = info['results']
        assert len(results) == len(self.dataset.seqs), "{} vs {}".format(len(results), len(self.dataset.seqs))
        if len(self.dataset.seqs[0].gt_rects) == 1:
            print("No GT annotations. Skip evaluation！")
            return dict(name='unknown')
        all_ious = []
        for seq_id, seq in enumerate(self.dataset.seqs):
            gt_boxes = seq.gt_rects  # [N, 4] in [x1, y1, w, h]
            num_gts = len(gt_boxes)
            assert num_gts > 1, "Cannot find valid ground-truth annotations " \
                                "in {} ({})".format(seq.name, info['dataset_name'])
            res_boxes = results[seq_id]['track_boxes']
            img_height = results[seq_id]['height']
            img_width = results[seq_id]['width']
            if num_gts < len(res_boxes):
                res_boxes = res_boxes[:num_gts]
            assert len(res_boxes) == len(gt_boxes), "The size of tracking result should be " \
                                                    "equal to ground-truth. {} vs. {}". \
                format(len(res_boxes), len(gt_boxes))

            seq_ious = rect_iou(res_boxes[1:], gt_boxes[1:], bound=(img_width, img_height))

            if hasattr(seq, 'cover'):
                # print(len(seq.cover), len(seq_ious), seq.name)
                seq_ious = seq_ious[seq.cover[1:] > 0]

            all_ious.append(seq_ious)
        all_ious = np.concatenate(all_ious, axis=0)
        report = dict(
            ao=np.mean(all_ious),
            sr50=np.mean(all_ious > 0.5),
            sr75=np.mean(all_ious > 0.75)
        )

        if 'name' in info:
            report['name'] = info['name']

        return report


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T
