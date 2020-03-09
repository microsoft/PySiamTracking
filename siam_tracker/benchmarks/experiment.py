# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import json
import copy
from datetime import date

from .utils import mkdir, list2csv
from .builder import build_benchmark, build_evaluator
from .hyperparam_iterator import HyperParamIterator
from ..models import build_tracker
from ..utils import load_checkpoint


def run_experiment(model_cfg: dict,
                   test_cfg: dict,
                   eval_cfg: dict,
                   checkpoint: str,
                   work_dir: str = '',
                   hypersearch: bool = False,
                   **kwargs):

    logging.getLogger()

    # build testing dataset
    eval_cfg = eval_cfg.copy()
    if 'hypers' in eval_cfg:
        hyper_cfg = eval_cfg.pop('hypers')
    else:
        hyper_cfg = None

    benchmark = build_benchmark(eval_cfg['dataset'])
    evaluator = build_evaluator(eval_cfg['metrics'])
    evaluator.register_dataset(benchmark)

    if work_dir == '':
        if os.path.isdir(checkpoint):
            work_dir = os.path.join(checkpoint, 'test_{}'.format(benchmark.name))
        else:
            dir_name = os.path.dirname(checkpoint)
            base_name = os.path.basename(checkpoint)[:-4]  # '*.pth'
            work_dir = os.path.join(dir_name, 'test_{}'.format(benchmark.name), base_name)

    if hypersearch:
        assert os.path.isdir(checkpoint), 'when hyperparameter searching is enabled, the ' \
                                          'checkpoint must be a directory. {}'.format(checkpoint)
        if 'epoch' in hyper_cfg:
            epoch_list = hyper_cfg.pop('epoch')
            epoch_list = ['epoch_{}.pth'.format(epoch) for epoch in epoch_list]
        else:
            epoch_list = ['latest.pth']
        epoch_model_list = [os.path.join(checkpoint, epoch) for epoch in epoch_list]

        hyper_iterator = HyperParamIterator(hyper_cfg, test_cfg)
        test_cfg_list = []
        checkpoint_list = []
        work_dir_list = []
        for epoch_model in epoch_model_list:
            for _test_cfg, values in hyper_iterator:
                epoch_name = '.'.join(os.path.basename(epoch_model).split('.')[:-1])
                exp_name = '-'.join(['{}@{}'.format(v[0], v[1]) for v in values])
                exp_name = '{}-{}'.format(epoch_name, exp_name)
                i_work_dir = os.path.join(work_dir, exp_name)
                test_cfg_list.append(copy.deepcopy(_test_cfg))
                checkpoint_list.append(epoch_model)
                work_dir_list.append(i_work_dir)

    else:
        assert os.path.exists(checkpoint), 'cannot find checkpoint {}'.format(checkpoint)
        test_cfg_list = [test_cfg]
        checkpoint_list = [checkpoint]
        work_dir_list = [work_dir]

    report_list = []
    logging.info("All {} experiments".format(len(test_cfg_list)))
    for i in range(len(test_cfg_list)):
        logging.info("Testing {} [{}/{}]".format(work_dir_list[i], i+1, len(test_cfg_list)))
        mkdir(work_dir_list[i])
        with open(os.path.join(work_dir_list[i], 'test_cfg.json'), 'w') as f:
            json.dump(test_cfg_list[i], f, indent='\t')
        # build a tracker w.r.t test_cfg & checkpoint
        tracker = build_tracker(model_cfg, test_cfg=test_cfg_list[i], is_training=False)
        load_checkpoint(tracker, checkpoint_list[i])
        # build a evaluator
        dataset_result = evaluator.run_dataset(tracker=tracker,
                                               dataset=benchmark,
                                               work_dir=work_dir_list[i], **kwargs)
        report = evaluator.evaluate(dataset_result, dataset=benchmark)
        report['name'] = os.path.basename(work_dir_list[i])
        logging.info('------------')
        for k, v in report.items():
            logging.info('{}: {}'.format(k, v))
        logging.info('------------')
        report_list.append(report)
    list2csv(report_list, os.path.join(work_dir, 'report.csv'))
    list2csv(report_list, os.path.join(work_dir, 'report_{}.csv'.format(date.today().strftime("%Y%m%d"))))
    return report_list
