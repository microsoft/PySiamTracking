# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pickle
from typing import Callable


def read_file(file_path: str, func: Callable = None):
    assert os.path.exists(file_path), 'Cannot find {}'.format(file_path)
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    out = []
    for line in lines:
        if line.strip() == '':
            continue
        if func is not None:
            out.append(func(line))
        else:
            out.append(line)
    return out


def mkdir(dir_path: str):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def read_pkl(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, 'rb') as f:
            ret = pickle.load(f)
    except Exception:
        return None
    return ret


def write_pkl(obj, pkl_path):
    pkl_dir = os.path.dirname(pkl_path)
    mkdir(pkl_dir)
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception:
        return False
    return True
