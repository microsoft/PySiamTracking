import numpy as np
import torch
import time
import os

from .base_evaluator import BaseEvaluator
from ..utils import DataPool, read_pkl, write_pkl
from ..utils.vot import (get_axis_aligned_bbox, vot_overlap, calculate_accuracy, calculate_failures,
                         calculate_expected_overlap)
from ..builder import EVALUATORS
from ...utils import box as ubox


@EVALUATORS.register_module
class Restart(BaseEvaluator):

    def __init__(self,
                 skipping=5,
                 low=100,
                 high=356,
                 peak=160):
        super(Restart, self).__init__(eval_type='restart')
        self.skipping = skipping
        self.low = low
        self.high = high
        self.peak = peak

    def evaluate(self,
                 dataset_result,
                 dataset=None,
                 tag='all'):
        if dataset is None:
            dataset = self.dataset
        results = dataset_result['results']
        assert len(results) == len(dataset.seqs), "{} vs {}".format(len(results), len(dataset.seqs))

        all_overlaps = []
        all_failures = []
        video_names = []
        gt_traj_length = []
        for seq_id, seq in enumerate(dataset.seqs):
            gt_traj = seq.gt_rects
            tracker_traj = results[seq_id]['pred_bboxes']
            tracker_traj = [t if isinstance(t, (list, tuple)) else [t] for t in tracker_traj]
            gt_traj_length.append(len(gt_traj))
            video_names.append(seq.name)
            overlaps = calculate_accuracy(tracker_traj, gt_traj,
                                          bound=(results[seq_id]['width'] - 1,
                                                 results[seq_id]['height'] - 1))[1]
            failures = calculate_failures(tracker_traj)[1]
            all_overlaps.append(overlaps)
            all_failures.append(failures)
        fragment_num = sum([len(x) + 1 for x in all_failures])
        max_len = max([len(x) for x in all_overlaps])
        seq_weight = 1.0

        # prepare segments
        fweights = np.ones((fragment_num)) * np.nan
        fragments = np.ones((fragment_num, max_len)) * np.nan
        seg_counter = 0
        for name, traj_len, failures, overlaps in zip(video_names, gt_traj_length,
                                                      all_failures, all_overlaps):
            if len(failures) > 0:
                points = [x + self.skipping for x in failures if
                          x + self.skipping <= len(overlaps)]
                points.insert(0, 0)
                for i in range(len(points)):
                    if i != len(points) - 1:
                        fragment = np.array(overlaps[points[i]:points[i + 1] + 1])
                        fragments[seg_counter, :] = 0
                    else:
                        fragment = np.array(overlaps[points[i]:])
                    fragment[np.isnan(fragment)] = 0
                    fragments[seg_counter, :len(fragment)] = fragment
                    if i != len(points) - 1:
                        # tag_value = dataset[name].tags[tag][points[i]:points[i+1]+1]
                        tag_value = dataset.select_tag(name, tag, points[i], points[i + 1] + 1)
                        w = sum(tag_value) / (points[i + 1] - points[i] + 1)
                        fweights[seg_counter] = seq_weight * w
                    else:
                        # tag_value = dataset[name].tags[tag][points[i]:len(overlaps)]
                        tag_value = dataset.select_tag(name, tag, points[i], len(overlaps))
                        w = sum(tag_value) / (traj_len - points[i] + 1e-16)
                        fweights[seg_counter] = seq_weight * w  # (len(fragment) / (traj_len-points[i]))
                    seg_counter += 1
            else:
                # no failure
                max_idx = min(len(overlaps), max_len)
                fragments[seg_counter, :max_idx] = overlaps[:max_idx]
                # tag_value = dataset[name].tags[tag][:max_idx]
                tag_value = dataset.select_tag(name, tag, 0, max_idx)
                w = sum(tag_value) / max_idx
                fweights[seg_counter] = seq_weight * w
                seg_counter += 1

        expected_overlaps = calculate_expected_overlap(fragments, fweights)
        # caculate eao
        weight = np.zeros((len(expected_overlaps)))
        weight[self.low - 1:self.high - 1 + 1] = 1
        is_valid = np.logical_not(np.isnan(expected_overlaps))
        eao = np.sum(expected_overlaps[is_valid] * weight[is_valid]) / np.sum(weight[is_valid])

        # raw statistics to tracking metrics
        report = dict(
            eao=eao,
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
                     work_dir: str = ''):
        # Try to load from files.
        if work_dir != '':
            saved_path = os.path.join(work_dir, '{}_tmp.pkl'.format(sequence.name))
            result = read_pkl(saved_path)
            if result is not None:
                return result

        data_pool = DataPool(sequence, prefetch=prefetch)

        frame_counter = 0
        lost_number = 0
        pred_bboxes = []

        start = time.time()
        duration = 0.0
        img_height, img_width = None, None
        for idx in range(len(sequence)):
            tic = time.time()
            img = data_pool[idx]
            if use_gpu:
                img = img.cuda()
            if img_height is None:
                img_height, img_width = img.size(2), img.size(3)
            gt_bbox = sequence.gt_rects[idx]
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1] + gt_bbox[3],
                           gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3],
                           gt_bbox[0] + gt_bbox[2], gt_bbox[1]]
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx, cy, w, h]
                init_bbox = torch.FloatTensor(gt_bbox_).view(1, 4)
                tracker.initialize(img, init_bbox)
                pred_bbox = gt_bbox_
                pred_bboxes.append(1)
            elif idx > frame_counter:
                pred_bbox = tracker.predict(img)
                pred_bbox = pred_bbox.cpu().numpy().reshape(-1)
                pred_bbox = ubox.xcycwh_to_xywh(pred_bbox).tolist()
                overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[3], img.shape[2]))
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5  # skip 5 frames
                    lost_number += 1
            else:
                pred_bboxes.append(0)
            duration += (time.time() - tic)

        result = dict(
            height=img_height,
            width=img_width,
            pred_bboxes=pred_bboxes,
            duration=duration,
            total_time=(time.time() - start),
            nframes=len(sequence),
            lost_number=lost_number
        )
        if work_dir != '':
            write_pkl(result, saved_path)

        return result
