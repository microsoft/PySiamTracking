# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import logging
from torch import multiprocessing


def single_test(evaluator: BaseEvaluator,
                tracker: BaseTracker,
                dataset: Dataset,
                gpus: int,
                prefetch: bool = False):
    logger = logging.getLogger()
    use_gpu = gpus > 0
    tracker.eval()
    if use_gpu:
        tracker.cuda()

    results = []

    for seq in dataset:
        i_result = evaluator.run_sequence(tracker,
                                          seq,
                                          use_gpu=use_gpu,
                                          zero_based_index=dataset.zero_based_index,
                                          prefetch=prefetch)
        results.append(i_result)
        logger.info("Test '{}' in {:.2f}s".format(seq.name, i_result['total_time']))
    return dict(
        dataset_type=str(type(dataset).__name__),
        eval_type=str(type(evaluator).__name__),
        results=results
    )


def parallel_test(evaluator: BaseEvaluator,
                  tracker: BaseTracker,
                  dataset: Dataset,
                  gpus: int,
                  prefetch: bool = False,
                  workers_per_gpu=1):
    logger = logging.getLogger()
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = gpus * workers_per_gpu

    # duplicate the tracker for 'num_workers' times.
    tracker_list = []
    with torch.no_grad():
        for i in range(num_workers):
            # copy module
            _tracker = copy.deepcopy(tracker)
            # # copy state dict
            new_state_dict = OrderedDict()
            for k, v in tracker.state_dict().items():
                new_state_dict[k] = v.clone().cpu()
            _tracker.load_state_dict(new_state_dict)
            tracker_list.append(_tracker)

