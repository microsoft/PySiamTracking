# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from . import assigners, samplers


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        args = cfg.copy()
        obj_type = args.pop('type')
        obj_type = getattr(assigners, obj_type)
        return obj_type(**args)
    else:
        raise TypeError('Invalid type {} for building a assigner'.format(type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        args = cfg.copy()
        obj_type = args.pop('type')
        obj_type = getattr(samplers, obj_type)
        return obj_type(**args)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes)
    return assign_result, sampling_result
