# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.


import torch
import logging
from torch.nn.utils import clip_grad

from .hook import Hook


def group_single_model(model):
    """ Group model parameters """
    # find all the BN layers
    bn_layer_name = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_layer_name.append(name)

    param_list = []
    for name, p in model.named_parameters():
        is_frozen = False if p.requires_grad else True
        is_bn = False
        for bn_name in bn_layer_name:
            if name.startswith(bn_name):
                is_bn = True
                break
        is_bias = True if 'bias' in name else False
        is_backbone = True if 'z_net.' in name else False

        param_list.append(dict(
            is_bn=is_bn,
            is_frozen=is_frozen,
            is_backbone=is_backbone,
            is_bias=is_bias,
            name=name,
            param=p
        ))

    return param_list


def collect_params(model, param_type):

    param_attr_list = group_single_model(model)

    names = []
    params = []

    for param_attr in param_attr_list:
        is_target = False
        if param_type == 'all':
            is_target = True
        elif param_type == 'all/weight':
            if not param_attr['is_bias'] and not param_attr['is_bn']:
                is_target = True
        elif param_type == 'all/bias':
            if param_attr['is_bias'] or param_attr['is_bn']:
                is_target = True
        elif param_type == 'head':
            if not param_attr['is_backbone']:
                is_target = True
        elif param_type == 'head/weight':
            if not param_attr['is_backbone']:
                if not param_attr['is_bias'] and not param_attr['is_bn']:
                    is_target = True
        elif param_type == 'head/bias':
            if not param_attr['is_backbone']:
                if param_attr['is_bias'] or param_attr['is_bn']:
                    is_target = True
        elif param_type == 'backbone':
            if param_attr['is_backbone']:
                is_target = True
        elif param_type == 'backbone/weight':
            if param_attr['is_backbone']:
                if not param_attr['is_bias'] and not param_attr['is_bn']:
                    is_target = True
        elif param_type == 'backbone/bias':
            if param_attr['is_backbone']:
                if param_attr['is_bias'] or param_attr['is_bn']:
                    is_target = True
        else:
            raise NotImplementedError("Unknown parameter type {}".format(param_type))

        if is_target:
            names.append(param_attr['name'])
            params.append(param_attr['param'])

    return names, params


def build_param_groups(model, params_cfgs):
    if not isinstance(params_cfgs, (list, tuple)):
        params_cfgs = [params_cfgs]
    groups = []
    for params_cfg in params_cfgs:
        if isinstance(params_cfg, str):
            names, params = collect_params(model, params_cfg)
            group = {'params': params}
        elif isinstance(params_cfg, dict):
            names, params = collect_params(model, params_cfg['name'])
            group = {'params': params}
            for k, v in params_cfg.items():
                group[k] = v
        else:
            raise TypeError("Unknown type {}".format(type(params_cfg)))
        groups.append(group)

    return groups


def build_optimizer(model, optimizer_cfg):
    args = optimizer_cfg.copy()
    param_cfgs = args.pop('params')
    groups = build_param_groups(model, param_cfgs)
    optimizer_type = args.pop('type')
    _opt = getattr(torch.optim, optimizer_type)(params=groups, **args)
    return _opt


class OptimizerHook(Hook):

    def __init__(self,
                 grad_clip=None,
                 optimizer_cfg=None,
                 optimizer_schedule=None):

        self.optimizer_cfg = optimizer_cfg
        self.optimizer_schedule = optimizer_schedule
        self.grad_clip = grad_clip

    def before_run(self, runner):
        logger = logging.getLogger()
        cfg = None
        logger.info("Runner status: ")
        logger.info("Epoch {} iter {}".format(runner.epoch, runner.iter))
        if runner.epoch > 0 and self.optimizer_schedule is not None:
            # setup optimizer for the nearest epoch
            nearest_dist = 100000
            for opt_cfg in self.optimizer_schedule:
                start_epoch = opt_cfg['start_epoch']
                dist = runner.epoch - start_epoch
                if dist > 0 and dist < nearest_dist:
                    nearest_dist = dist
                    cfg = opt_cfg.copy()
            if cfg is not None:
                start_epoch = cfg.pop('start_epoch')
                logger.info("Load optimizer from Epoch [{}]".format(start_epoch))

        if cfg is None and self.optimizer_cfg is not None:
            cfg = self.optimizer_cfg

        if cfg is not None:
            logger.info('Build optimizer:')
            logger.info(cfg)
            # build a new optimizer
            runner.optimizer = build_optimizer(runner.model, cfg)

    def before_epoch(self, runner):
        logger = logging.getLogger()
        if self.optimizer_schedule is not None:
            # setup optimizer for current epoch
            for opt_cfg in self.optimizer_schedule:
                _cfg = opt_cfg.copy()
                start_epoch = _cfg.pop('start_epoch')
                if start_epoch == runner.epoch:
                    logger.info("Build optimizer:")
                    logger.info(opt_cfg)
                    runner.optimizer = build_optimizer(runner.model, _cfg)

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
