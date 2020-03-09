# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths
import os
import argparse
import logging

from siam_tracker.utils import Config
from siam_tracker.benchmarks.experiment import run_experiment


BENCHMARK_EVAL_TYPE = dict(
    VOT16='Restart',
    VOT17='Restart',
    OTB100='OPE',
    OTB50='OPE',
    OTB2013='OPE',
    LaSOTTest='OPE',
    LaSOTFull='OPE',
    GOT10KTest='GOTOPE',
    GOT10KVal='GOTOPE',
    TrackingNet='OPE',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a tracker.')
    parser.add_argument('--config',
                        default='',
                        type=str,
                        help='tracker configuration file path.')
    parser.add_argument('--benchmark',
                        default='',
                        type=str,
                        help='benchmark name, delimiter by comma. Current support benchmarks:' + \
                             ','.join(BENCHMARK_EVAL_TYPE.keys()))
    parser.add_argument('--eval_config',
                        default='',
                        type=str,
                        help='evaluation configuration file path, which specifies the benchmark and'
                             'evaluation metrics. If it is empty, we simply test the benchmarks given '
                             'by --benchmark field')
    parser.add_argument('--data_dir',
                        default='data/benchmark',
                        type=str,
                        help='benchmark data directory path. By default it is "data/benchmark".')
    parser.add_argument('--checkpoint',
                        default='',
                        type=str,
                        help='checkpoint path. If it is empty, we will search the path defined in '
                             'tracker configuration file.')
    parser.add_argument('--output_dir',
                        default='',
                        type=str,
                        help='output directory. If it is empty, the directory in checkpoint path will '
                             'be used.')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='number of GPUs. 0 means CPU testing. Any '
                             'value > 1 will lead to multiprocessing testing.')
    parser.add_argument('--hypersearch',
                        action='store_true',
                        help='enable hyper parameter search. If specified, the --eval_config should '
                             'not be empty.')
    parser.add_argument('--from_zip',
                        type=bool,
                        default=False,
                        help='loading dataset from zip file')
    parser.add_argument('opts',
                        help='all test options',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    args = parse_args()

    # setup model configuration and test configuration
    cfg = Config.fromfile(args.config)
    # disable loading pretrained backbone.
    if hasattr(cfg, 'model'):
        if hasattr(cfg.model, 'backbone'):
            cfg.model.backbone.pretrained = None
    # if we input optional configurations, we will override the original one.
    if len(args.opts) > 0:
        assert len(args.opts) % 2 == 0
        for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
            cfg.update_value(full_key, v)
            logging.info("Set {} --> {}".format(full_key, v))

    # setup evaluation configuration.
    if args.eval_config != '':
        eval_cfgs = Config.fromfile(args.eval_config).eval_cfgs
        for i in range(len(eval_cfgs)):
            eval_cfgs[i]['dataset']['data_root'] = args.data_dir
            eval_cfgs[i]['dataset']['zip_mode'] = args.from_zip
    elif args.benchmark != '':
        benchmark_list = args.benchmark.split(',')
        eval_cfgs = []
        for benchmark_name in benchmark_list:
            eval_cfgs.append(dict(
                metrics=dict(type=BENCHMARK_EVAL_TYPE[benchmark_name]),
                dataset=dict(type=benchmark_name, data_root=args.data_dir, zip_mode=args.from_zip)
            ))
    else:
        raise ValueError("No benchmark or evaluation configuration.")

    # setup checkpoint path
    if args.checkpoint == '':
        if args.hypersearch:
            checkpoint = cfg.work_dir
        else:
            checkpoint = os.path.join(cfg.work_dir, 'latest.pth')
    else:
        checkpoint = args.checkpoint

    # setup output directory
    if args.output_dir == '':
        if os.path.isdir(checkpoint):
            output_dir = checkpoint
        else:
            output_dir = os.path.dirname(checkpoint)
    else:
        output_dir = args.output_dir

    for eval_cfg in eval_cfgs:
        work_dir = os.path.join(output_dir, 'test_{}'.format(eval_cfg['dataset']['type'].lower()))
        if not args.hypersearch:
            suffix = os.path.basename(checkpoint)[:-4]
            if len(args.opts) > 0:
                assert len(args.opts) % 2 == 0
                for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
                    suffix = suffix + '-{}@{}'.format(full_key, v)
            work_dir = os.path.join(work_dir, suffix)

        run_experiment(model_cfg=cfg.model,
                       test_cfg=cfg.test_cfg,
                       eval_cfg=eval_cfg,
                       checkpoint=checkpoint,
                       work_dir=work_dir,
                       hypersearch=args.hypersearch,
                       gpus=args.gpus)
