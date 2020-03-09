import numpy as np
import torch
import time
import os

from .base_evaluator import BaseEvaluator
from ..utils import DataPool, read_pkl, write_pkl
from ..builder import EVALUATORS
from ...utils import box as ubox


@EVALUATORS.register_module
class OPE(BaseEvaluator):

    def __init__(self,
                 overlap_threshold=np.arange(0.0, 1.01, 0.05),
                 overlap_rank_index=10,
                 error_threshold=np.arange(0.0, 50.1, 1.0),
                 error_norm_threshold=np.arange(0.0, 0.501, 0.01),
                 error_rank_index=20):
        super(OPE, self).__init__(eval_type='ope')
        self.overlap_threshold = overlap_threshold
        self.overlap_rank_index = overlap_rank_index
        self.error_threshold = error_threshold
        self.error_norm_threshold = error_norm_threshold
        self.error_rank_index = error_rank_index

    def evaluate(self,
                 dataset_result,
                 dataset=None):
        if dataset is None:
            dataset = self.dataset
        seq_results = dataset_result['results']
        assert len(seq_results) == len(dataset.seqs), "{} vs {}".format(len(seq_results), len(dataset.seqs))
        if len(dataset.seqs[0].gt_rects) == 1:
            return dict(name='unknown')

        raw_stats = []
        total_time, total_time_wo_io, total_frame_count = 0.0, 0.0, 0
        for seq_id, seq in enumerate(dataset.seqs):
            gt_boxes = seq.gt_rects  # [N, 4] in [x1, y1, w, h]
            num_gts = len(gt_boxes)
            res_boxes = seq_results[seq_id]['track_boxes']
            total_frame_count += len(res_boxes)
            total_time += seq_results[seq_id]['total_time']
            total_time_wo_io += seq_results[seq_id]['duration']
            if num_gts < len(res_boxes):
                res_boxes = res_boxes[:num_gts]
            assert len(res_boxes) == len(gt_boxes), "The size of tracking result should be " \
                                                    "equal to ground-truth. {} vs. {}". \
                format(len(res_boxes), len(gt_boxes))
            if hasattr(seq, 'absence'):
                err_ctr, err_ctr_norm, err_coverage = cal_seq_error_robust(res_boxes, gt_boxes, seq.absence)
            else:
                err_ctr, err_ctr_norm, err_coverage = cal_seq_error_robust(res_boxes, gt_boxes, None)
            success_num_overlap = np.sum((err_coverage[:, np.newaxis] > self.overlap_threshold[np.newaxis, :]), axis=0)
            success_num_err = np.sum((err_ctr[:, np.newaxis] <= self.error_threshold[np.newaxis, :]), axis=0)
            success_num_err_norm = np.sum((err_ctr_norm[:, np.newaxis] <= self.error_norm_threshold[np.newaxis, :]), axis=0)
            raw_stat = dict(
                name=seq.name,
                err_ctr=err_ctr,
                err_ctr_norm=err_ctr_norm,
                err_coverage=err_coverage,
                success_num_err=success_num_err / num_gts,
                success_num_err_norm=success_num_err_norm / num_gts,
                success_num_overlap=success_num_overlap / num_gts
            )
            raw_stats.append(raw_stat)

        # raw statistics to tracking metrics
        report = dict(
            auc_overlap=self.auc(raw_stats, metric_type='overlap'),
            precision_error=self.threshold(raw_stats, metric_type='error'),
            precision_error_norm=self.threshold(raw_stats, metric_type='error_norm'),
            fps=total_frame_count / total_time,
            fps_wo_io=total_frame_count / total_time_wo_io,
        )
        if 'name' in dataset_result:
            report['name'] = dataset_result['name']

        return report

    def run_sequence(self,
                     tracker,
                     sequence,
                     use_gpu: bool = True,
                     zero_based_index: bool = True,
                     prefetch: bool = False,
                     work_dir: str = '') -> dict:
        # Try to load from files.
        if work_dir != '':
            saved_path = os.path.join(work_dir, '{}_tmp.pkl'.format(sequence.name))
            result = read_pkl(saved_path)
            if result is not None:
                return result

        data_pool = DataPool(sequence, prefetch=prefetch)
        gt_rects = torch.FloatTensor(sequence.gt_rects).clone()
        assert gt_rects.dim() == 2 and gt_rects.size(1) == 4, "The box should be in shape of [N, 4]."
        if not zero_based_index:
            gt_rects[:, 0:2] = gt_rects[:, 0:2] - 1
        gt_rects = ubox.xywh_to_xcycwh(gt_rects)
        start = time.time()
        duration = 0.0
        img_height, img_width = None, None

        for i, frame in enumerate(data_pool):
            if img_height is None:
                img_height, img_width = frame.size(2), frame.size(3)
            tic = time.time()
            if use_gpu:
                frame = frame.cuda()  # move image data to gpu device
            if i == 0:
                tracker.initialize(frame, gt_rects[0:1])
            else:
                tracker.predict(frame)
            duration += (time.time() - tic)
        track_boxes = torch.cat(tracker.tracking_results, dim=0).numpy()
        track_boxes = ubox.xcycwh_to_xywh(track_boxes)
        if not zero_based_index:
            track_boxes[:, 0:2] = track_boxes[:, 0:2] + 1
        result = dict(
            height=img_height,
            width=img_width,
            track_boxes=track_boxes,
            duration=duration,
            total_time=(time.time() - start),
            nframes=len(data_pool)
        )

        if work_dir != '':
            write_pkl(result, saved_path)

        return result

    def auc(self, raw_stats, metric_type='overlap'):
        success_rate = self._get_success_rate(raw_stats, metric_type=metric_type)
        return np.mean(success_rate)

    def threshold(self, raw_stats, metric_type='error'):
        success_rate = self._get_success_rate(raw_stats, metric_type=metric_type)
        if metric_type == 'overlap':
            idx = self.overlap_rank_index
        elif metric_type == 'error':
            idx = self.error_rank_index
        elif metric_type == 'error_norm':
            idx = self.error_rank_index
        else:
            raise ValueError("Unknown metric type {}".format(metric_type))
        return success_rate[idx]

    def _get_success_rate(self, raw_stats, metric_type):
        if metric_type == 'overlap':
            threshold_list = self.overlap_threshold
            key = 'success_num_overlap'
        elif metric_type == 'error':
            threshold_list = self.error_threshold
            key = 'success_num_err'
        elif metric_type == 'error_norm':
            threshold_list = self.error_norm_threshold
            key = 'success_num_err_norm'
        else:
            raise ValueError("Unknown metric type {}".format(metric_type))
        success_rate = average_success_rate(raw_stats, key, threshold_list)
        return success_rate


def average_success_rate(raw_stats, key, threshold_list):
    avg_success_rate = np.zeros((len(threshold_list), ), np.float64)
    for idx, raw_stat in enumerate(raw_stats):
        avg_success_rate = avg_success_rate + raw_stat[key]
    if len(raw_stats) > 0:
        avg_success_rate = avg_success_rate / len(raw_stats)
    return avg_success_rate


def cal_seq_error_robust(res_boxes, gt_boxes, absence=None):
    """ Calculate the error information in echo sequence.
    Same as 'calcSeqErrRobust' in offcial Matlab code
    """
    # convert to numpy.ndarry
    res_boxes = np.array(res_boxes, dtype=np.float64)
    gt_boxes = np.array(gt_boxes, dtype=np.float64)

    for i in range(1, res_boxes.shape[0]):
        r = res_boxes[i]
        if np.any(np.isnan(r)) or r[2] <= 0 or r[3] <= 0:
            res_boxes[i, :] = res_boxes[i-1, :]

    # remove absence frames if necessary
    if absence is not None:
        absence = np.array(absence, dtype=np.bool)
        valid_inds = np.where(np.logical_not(absence))[0]
        res_boxes = res_boxes[valid_inds]
        gt_boxes = gt_boxes[valid_inds]

    gt_ctr = _get_box_center(gt_boxes)

    # the first frame will always be correct
    res_boxes[0, :] = gt_boxes[0, :]
    res_ctr = _get_box_center(res_boxes)

    norm_term = np.clip(gt_boxes[:, 2:4], a_min=np.finfo(np.float64).eps, a_max=np.inf)
    res_ctr_norm = res_ctr / norm_term
    gt_ctr_norm = gt_ctr / norm_term
    delta_ctr = (res_ctr - gt_ctr)
    delta_ctr_norm = res_ctr_norm - gt_ctr_norm

    err_ctr = np.sqrt(np.sum(delta_ctr**2, axis=1))
    err_ctr_norm = np.sqrt(np.sum(delta_ctr_norm**2, axis=1))
    err_coverage = _overlap(res_boxes, gt_boxes)

    valid_gt_inds = np.all(gt_boxes > 0, axis=1)
    err_ctr[np.logical_not(valid_gt_inds)] = -1
    err_ctr_norm[np.logical_not(valid_gt_inds)] = -1
    err_coverage[np.logical_not(valid_gt_inds)] = -1

    return err_ctr, err_ctr_norm, err_coverage


# TODO(guangting): collect utilization functions into one file?
def _get_box_center(boxes):
    ctr_x = boxes[:, 0] + (boxes[:, 2] - 1.0) / 2.0
    ctr_y = boxes[:, 1] + (boxes[:, 3] - 1.0) / 2.0
    ctr = np.hstack((ctr_x[:, np.newaxis], ctr_y[:, np.newaxis]))
    return ctr


def _xyxy_to_xywh(xyxy):
    assert isinstance(xyxy, np.ndarray)
    xywh = xyxy.copy()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0] + 1.0
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1] + 1.0


def _xywh_to_xyxy(xywh):
    assert isinstance(xywh, np.ndarray)
    xyxy = xywh.copy()
    xyxy[:, 2] = xyxy[:, 2] + xyxy[:, 0] - 1.0
    xyxy[:, 3] = xyxy[:, 3] + xyxy[:, 1] - 1.0
    return xyxy


def _overlap(a, b):
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]

    a = _xywh_to_xyxy(a)
    b = _xywh_to_xyxy(b)
    ix1 = np.maximum(a[:, 0], b[:, 0])
    iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2])
    iy2 = np.minimum(a[:, 3], b[:, 3])
    iw = np.maximum(0.0, ix2 - ix1 + 1.0)
    ih = np.maximum(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih

    return inter / (area_a + area_b - inter)
