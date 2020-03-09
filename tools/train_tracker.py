# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths
import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.cuda

from siam_tracker.models import build_train_wrapper
from siam_tracker.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a siamese-network based tracker.')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--data_dir', help='the dir that save training data')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--debug', action="store_true", help='debug mode')
    parser.add_argument('--from_file', action="store_true", help='storage backend == file')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    args = parser.parse_args()

    return args


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':

    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.data_dir is not None:
        cfg.data_root = args.data_dir
        for i in range(len(cfg.train_cfg.train_data.datasets)):
            cfg.train_cfg.train_data.datasets[i].data_root = args.data_dir
    if args.debug:
        cfg.train_cfg.train_data.datasets = cfg.train_cfg.train_data.datasets[:1]
        cfg.train_cfg.train_data.datasets[0].name = 'got10k_mini'
        cfg.train_cfg.train_data.datasets[0].storage = dict(type='ZipWrapper', cache_into_memory=True)
        cfg.train_cfg.train_data.datasets[0].sample_weight = 1.0
        cfg.train_cfg.log_config.interval = 1
    if args.from_file:
        for i in range(len(cfg.train_data.datasets)):
            cfg.train_cfg.train_data.datasets[i].storage = dict(type='FileWrapper')

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if cfg.train_cfg.checkpoint_config is not None:
        cfg.train_cfg.checkpoint_config.meta = dict(config=cfg.text)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    train_wrapper = build_train_wrapper(train_cfg=cfg.train_cfg,
                                        model_cfg=cfg.model,
                                        work_dir=cfg.work_dir,
                                        log_level=cfg.log_level,
                                        resume_from=cfg.resume_from,
                                        gpus=cfg.gpus)

    train_wrapper.run()
