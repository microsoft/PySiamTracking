# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import zipfile
import io
import os


def build_storage(storage_cfg, root_path):
    _storage_cfg = storage_cfg.copy()
    storage_type = _storage_cfg.pop('type')
    t = getattr(sys.modules[__name__], storage_type)
    return t(root_path=root_path, **_storage_cfg)


class ZipWrapper(object):

    def __init__(self, root_path, cache_into_memory=False, **kwargs):

        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')

    def __getitem__(self, key):
        buf = self.zip_file.read(name=key)
        return buf

    def __del__(self):
        self.zip_file.close()


class FileWrapper(object):
    def __init__(self, root_path, **kwargs):
        self.root_path = root_path

    def __getitem__(self, key):
        with open(os.path.join(self.root_path, key), 'rb') as f:
            buf = f.read()
        return buf
