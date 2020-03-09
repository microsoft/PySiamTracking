# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from functools import partial
import torch
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

