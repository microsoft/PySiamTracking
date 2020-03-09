import os
import logging
import copy
from shutil import rmtree
from collections import OrderedDict
import torch
from torch import multiprocessing

from ..data import Dataset, Sequence
from ..utils import mkdir, read_pkl, write_pkl
from ...models.trackers.base_tracker import BaseTracker


class BaseEvaluator(object):

    def __init__(self, eval_type: str = ''):
        self.eval_type = eval_type
        self.dataset = None

    def register_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def evaluate(self,
                 dataset_result: dict,
                 dataset: Dataset = None) -> dict:
        """ Evaluate the performance on the target dataset. """
        raise NotImplementedError

    def run_sequence(self,
                     tracker: BaseTracker,
                     sequence: Sequence,
                     use_gpu: bool = True,
                     zero_based_index: bool = True,
                     prefetch: bool = False,
                     work_dir: str = '') -> dict:
        """ Run tracker on single sequence. """
        raise NotImplementedError

    def run_dataset(self,
                    tracker: BaseTracker,
                    dataset: Dataset = None,
                    gpus: int = 1,
                    work_dir: str = '',
                    prefetch: bool = False):
        """ Run tracker on a dataset. """
        tracker.cpu()
        logger = logging.getLogger()
        if dataset is None:
            dataset = self.dataset
        tmp_dir = ''
        if work_dir != '':
            mkdir(work_dir)
            tmp_dir = os.path.join(work_dir, 'tmp')
            mkdir(tmp_dir)
            save_path = os.path.join(work_dir, '{}_{}_results.pkl'.format(dataset.name, self.eval_type))
            # try to load from cached file
            results = read_pkl(save_path)
            if results is not None:
                logger.info("Loading file {}".format(save_path))
                return results

        if gpus <= 1:
            # single GPU testing.
            dataset_result = single_test(self, tracker, dataset, gpus, prefetch, tmp_dir)
        else:
            # multi GPU / multi processing testing.
            dataset_result = parallel_test(self, tracker, dataset, gpus, prefetch, tmp_dir)

        if work_dir != '':
            write_pkl(dataset_result, save_path)
            rmtree(tmp_dir, ignore_errors=True)
        return dataset_result


def single_test(evaluator: BaseEvaluator,
                tracker: BaseTracker,
                dataset: Dataset,
                gpus: int,
                prefetch: bool = False,
                work_dir: str = ''):
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
                                          prefetch=prefetch,
                                          work_dir=work_dir)
        results.append(i_result)
        logger.info("Test '{}' in {:.2f}s".format(seq.name, i_result['total_time']))
    return dict(
        dataset_type=str(type(dataset).__name__),
        eval_type=str(type(evaluator).__name__),
        results=results,
    )


def parallel_work_func(evaluator: BaseEvaluator,
                       tracker: BaseTracker,
                       dataset: Dataset,
                       gpu_id: int,
                       prefetch: bool,
                       work_dir: str,
                       idx_queue: multiprocessing.Queue,
                       result_queue: multiprocessing.Queue):
    torch.cuda.set_device(gpu_id)
    tracker.cuda()
    tracker.eval()
    while True:
        idx = idx_queue.get()
        seq = dataset[idx]
        i_result = evaluator.run_sequence(tracker,
                                          seq,
                                          use_gpu=True,
                                          zero_based_index=dataset.zero_based_index,
                                          prefetch=prefetch,
                                          work_dir=work_dir)
        result_queue.put((idx, i_result))


def parallel_test(evaluator: BaseEvaluator,
                  tracker: BaseTracker,
                  dataset: Dataset,
                  gpus: int,
                  prefetch: bool = False,
                  work_dir: str = '',
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

    gpu_id_list = [i % gpus for i in range(num_workers)]
    workers = [
        ctx.Process(
            target=parallel_work_func,
            args=(evaluator, tracker_list[i], dataset, gpu_id_list[i], prefetch, work_dir, idx_queue, result_queue)
        )
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
    for _ in range(len(dataset)):
        idx, res = result_queue.get()
        results[idx] = res
        logger.info('Test {} in {:.2f}s'.format(dataset.seqs[idx].name, res['total_time']))
    for worker in workers:
        worker.terminate()

    return dict(
        dataset_type=str(type(dataset).__name__),
        eval_type=str(type(evaluator).__name__),
        results=results
    )
