# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from collections import OrderedDict


def list2csv(contents, file_path, delimiter=','):
    """
    Args:
        contents (list[dict]): a list of dict
        file_path (str): file path
        delimiter (str): the delimiter in csv file.
    """
    assert delimiter in ('\t', ',')
    # extract all possible keys
    key_value_list = OrderedDict()
    if any(['name' in c for c in contents]):
        key_value_list['name'] = True

    for content in contents:
        assert isinstance(content, dict)
        for k in content.keys():
            if k not in key_value_list:
                key_value_list[k] = True

    output_dir = os.path.dirname(file_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(file_path, 'w') as f:
        for k in key_value_list.keys():
            f.write('{}{}'.format(k, delimiter))
        f.write('\n')
        for content in contents:
            for k in key_value_list.keys():
                if k not in content:
                    f.write('NULL{}'.format(delimiter))
                else:
                    if isinstance(content[k], float):
                        f.write('{:.5f}{}'.format(content[k], delimiter))
                    else:
                        f.write('{}{}'.format(content[k], delimiter))
            f.write('\n')
