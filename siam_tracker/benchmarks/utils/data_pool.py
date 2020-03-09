# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import zipfile
import io
import cv2
import torch
import os
import numpy as np


class DataPool(object):

    def __init__(self, sequence, prefetch=False):
        self.seq = sequence
        self.imgs = []
        self.prefetch = prefetch

        if self.seq.zip_content is not None:
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.seq.zip_content), 'r')
        else:
            self.zip_file = None

        if prefetch:
            for idx in range(len(sequence)):
                img_tensor = self.get_img_tensor(idx)
                self.imgs.append(img_tensor)

    def __getitem__(self, idx) -> torch.Tensor:
        if self.prefetch:
            return self.imgs[idx]
        else:
            return self.get_img_tensor(idx)

    def __len__(self):
        return len(self.seq)

    def get_img_tensor(self, idx) -> torch.Tensor:
        img_path = self.seq.frames[idx]
        if self.zip_file is not None:
            img_basename = os.path.basename(img_path)
            imgbuf = self.zip_file.read(name=img_basename)
            img = cv2.imdecode(
                np.fromstring(imgbuf, dtype=np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
            assert img is not None
        else:
            if not os.path.exists(img_path):
                raise FileNotFoundError("Cannot find image {}".format(img_path))
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        # convert to RGB [0, 1]
        img_tensor = img_tensor[[2, 1, 0], :, :].contiguous()
        img_tensor.div_(255.0)
        for i, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
            img_tensor[i].sub_(mean)
            img_tensor[i].div_(std)

        img_tensor.unsqueeze_(0)
        return img_tensor
