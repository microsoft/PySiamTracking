# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from itertools import product


class HyperParamIterator(object):

    def __init__(self, hyperparams, test_cfg):
        """ In visual tracking, it usually needs hyper-parameter searching.
        The 'hyperparams' is a dict. The keys are where to traverse and the values are
        list of candidates values.
        e.g.,

        test_cfg = {
            'post_processing': {
                'windows': 0.3,
            },
            'x_size': 255,
            ...
        }
        hyperparams = {
            'post_processing': {
                'windows': [0.2, 0.3, 0.4],
            },
            'x_size': [255, 271],
        }

        The iterator will generate and then return all the combinations of hyper-param list:
        [
            {
                'post_processing': {
                    'windows': 0.2,
                },
                ‘x_size': 255,
                ...
            },
            {
                'post_processing': {
                    'windows': 0.2,
                },
                ‘x_size': 271,
                ...
            },
            ...
        ]
        """

        self.test_cfg = test_cfg
        self.hyperparams = hyperparams

        self.key_list = _dictkey2list(hyperparams)
        candidate_list = [_get_item(hyperparams, k) for k in self.key_list]
        # all possible combinations in the search space.
        self.combinations = list(product(*candidate_list))

        self.index = 0

    def __len__(self):
        return len(self.combinations)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if len(self.combinations) == 0:
            if self.index == 0:
                self.index += 1
                return self.test_cfg, None
            else:
                raise StopIteration
        else:
            if self.index >= len(self.combinations):
                raise StopIteration
            else:
                comb = self.combinations[self.index]
                self.index += 1
                test_cfg = _merge_cfg(self.test_cfg, self.key_list, comb)
                return test_cfg, list(zip(self.key_list, comb))


def _merge_cfg(core_cfg, key_list, value_list):
    assert len(key_list) == len(value_list)
    _cfg = core_cfg.copy()
    for k, v in zip(key_list, value_list):
        _set_item(_cfg, k, v)
    return _cfg


def _dictkey2list(d, prefix=''):
    """ Save all leaf-node keys in the dict into a list.
    e.g.:
    d = {
        'a': [0],
        'b': {
            'c': [1]
        },
        'd': [2]
    }
    _dictkey2list(d) = ['a', 'b.c', 'd']
    """
    if d is None:
        return []

    key_list = []
    for k in d.keys():
        if isinstance(d[k], list):
            key_list.append(prefix + k)
        elif isinstance(d[k], dict):
            prefix_next = prefix + k + '.'
            key_list = key_list + _dictkey2list(d[k], prefix_next)
        else:
            raise TypeError("Unsupport type in {}: '{}'".format(k, type(d[k])))
    return key_list


def _get_item(d, k):
    """ Get item from dict according to a key. """
    split = k.split('.')
    if len(split) == 1:
        return d[split[0]]
    else:
        return _get_item(d[split[0]], '.'.join(split[1:]))


def _set_item(d, k, v):
    split = k.split('.')
    if len(split) == 1:
        d[split[0]] = v
    else:
        _set_item(d[split[0]], '.'.join(split[1:]), v)
