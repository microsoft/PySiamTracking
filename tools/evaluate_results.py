# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths
import os
import fnmatch
import pdb
import glob
import argparse
import pickle
import pandas as pd
import logging
from tabulate import tabulate


from siam_tracker.benchmarks import build_benchmark, build_evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate results')
    parser.add_argument('--result',
                        default='',
                        type=str,
                        help='result file path.')
    parser.add_argument('--data_dir',
                        default='data/benchmark',
                        type=str,
                        help='benchmark data directory path. By default it is "data/benchmark".')
    parser.add_argument('--sort_key',
                        default='',
                        type=str,
                        help='sort the result table by given key')
    parser.add_argument('--topk',
                        default=-1,
                        type=int,
                        help='keep top-K entries.')
    args = parser.parse_args()
    return args


def longest_common_prefix(strs_list):
    prefix = ''
    for _, item in enumerate(zip(*strs_list)):
        if len(set(item)) > 1:
            return len(prefix)
        else:
            prefix += item[0]
    return len(prefix)


def longest_common_suffix(strs_list):
    new_strs_list = [s[::-1] for s in strs_list]
    return longest_common_prefix(new_strs_list)


if __name__ == '__main__':
    args = parse_args()
    if ':' in args.result:
        result_list = args.result.split(':')
    elif os.path.isfile(args.result):
        result_list = [args.result]
    else:
        result_list = glob.glob(args.result, recursive=True)

    benchmark = None
    evaluator = None

    table = None
    metrics_keys = None

    if len(result_list) == 1:
        name_list = [os.path.basename(os.path.dirname(os.path.abspath(result_list[0])))]
    elif len(result_list) > 1:
        prefix_len = longest_common_prefix(result_list)
        suffix_len = longest_common_suffix(result_list)
        name_list = [result_path[prefix_len:-suffix_len] for result_path in result_list]
    else:
        print("No valid result file..")
        exit(0)
    for i, result_path in enumerate(result_list):
        assert result_path.endswith('.pkl'), result_path
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
        dataset_type = result['dataset_type']
        eval_type = result['eval_type']
        if benchmark is None:
            benchmark = build_benchmark(dict(type=dataset_type, data_root=args.data_dir))
            evaluator = build_evaluator(dict(type=eval_type))
        else:
            assert type(benchmark).__name__ == dataset_type
            assert type(evaluator).__name__ == eval_type
        report = evaluator.evaluate(result, benchmark)
        if metrics_keys is None:
            metrics_keys = [k for k in report.keys()]
            table = pd.DataFrame(columns=['name'] + metrics_keys)
        row = [name_list[i]] + [report[k] for k in metrics_keys]
        table.loc[i] = row
    if args.sort_key != '':
        table = table.sort_values(by=[args.sort_key], ascending=False)
    if args.topk > 0:
        table = table.head(args.topk)
    print(tabulate(table, headers='keys', tablefmt='psql'))
