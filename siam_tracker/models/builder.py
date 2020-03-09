# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from ..utils import Registry


BACKBONES = Registry(name='backbone')
TRACKERS = Registry(name='tracker')
HEADS = Registry(name='head')
NECKS = Registry(name='neck')
FUSIONS = Registry(name='fusion')
LOSSES = Registry(name='loss')

TRAIN_WRAPPERS = Registry(name='train_wrappers')


def build_backbone(cfg):
    _cfg = cfg.copy()
    backbone_type = _cfg.pop('type')
    return BACKBONES.get_module(backbone_type)(**_cfg)


def build_head(cfg):
    _cfg = cfg.copy()
    head_type = _cfg.pop('type')
    return HEADS.get_module(head_type)(**_cfg)


def build_neck(cfg):
    _cfg = cfg.copy()
    neck_type = _cfg.pop('type')
    return HEADS.get_module(neck_type)(**_cfg)


def build_loss(cfg):
    _cfg = cfg.copy()
    loss_type = _cfg.pop('type')
    return LOSSES.get_module(loss_type)(**_cfg)


def build_fusion(cfg):
    _cfg = cfg.copy()
    fusion_type = _cfg.pop('type')
    return FUSIONS.get_module(fusion_type)(**_cfg)


def build_tracker(cfg, test_cfg=None, train_cfg=None, is_training=False):
    _cfg = cfg.copy()
    tracker_type = _cfg.pop('type')
    tracker = TRACKERS.get_module(tracker_type)(**_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    tracker.set_phase(is_training=is_training)
    return tracker


def build_train_wrapper(train_cfg, *args, **kwargs):
    training_type = train_cfg['type']
    train_wrapper = TRAIN_WRAPPERS.get_module(training_type)(train_cfg=train_cfg, *args, **kwargs)
    return train_wrapper
