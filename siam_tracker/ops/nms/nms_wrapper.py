import numpy as np
import torch

from . import nms_cython_backend, nms_torch_backend


def nms(dets, iou_thr, return_inds=False):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    if dets.shape[0] == 0:
        if not return_inds:
            return dets
        else:
            return dets, [0]

    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        # sort
        inds = torch.argsort(dets[:, 4], descending=True)
        dets = dets[inds].contiguous()
        keep_inds = nms_torch_backend.forward(dets, float(iou_thr)).long()
        dets = dets[keep_inds].contiguous()
        keep_inds = inds[keep_inds]
    elif isinstance(dets, np.ndarray):
        keep_inds = nms_cython_backend.nms(dets, iou_thr)
        dets = dets[keep_inds]
        assert not return_inds
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))
    if return_inds:
        return dets, keep_inds
    else:
        return dets

