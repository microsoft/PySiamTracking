# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from ..utils import Registry


BENCHMARKS = Registry(name='benchmark')
EVALUATORS = Registry(name='evaluator')


def build_benchmark(cfg):
    _cfg = cfg.copy()
    benchmark_type = _cfg.pop('type')
    return BENCHMARKS.get_module(benchmark_type)(**_cfg)


def build_evaluator(cfg):
    _cfg = cfg.copy()
    benchmark_type = _cfg.pop('type')
    return EVALUATORS.get_module(benchmark_type)(**_cfg)
