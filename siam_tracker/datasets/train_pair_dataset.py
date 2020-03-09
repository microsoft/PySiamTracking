# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import time
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

from . import transforms
from .image_dataset import ImageDataset
from ..utils.parallel import DataContainer
from ..utils import img_norm, img_gray


class TrainPairDataset(Dataset):
    """ Sample pair of images for training a siamese-network"""
    def __init__(self, data_cfg):
        self.sample_per_epoch = data_cfg.sample_per_epoch
        self.gray_prob = data_cfg.gray_prob
        self._load_datasets(data_cfg.datasets)
        self._load_transforms(data_cfg.transforms)

    def __len__(self):
        return self.sample_per_epoch

    def __getitem__(self, item):
        meta_info = self.sample_from_dataset()
        out = self.meta_info_to_tensor(meta_info)
        return out

    def meta_info_to_tensor(self, meta_info):
        """ some necessary pre-processing for sampled image pairs """
        # extract some information from meta_info dict
        assert len(meta_info['img_list']) == 2, "Support two images for training only."
        z_img = meta_info['img_list'][0]
        z_box = meta_info['box_list'][0]

        x_img = meta_info['img_list'][1]
        x_box = meta_info['box_list'][1]
        x_flag = meta_info['flag_list'][1]

        # apply for data-augmentation
        z_transform = self.iz_trans if meta_info['is_image'] else self.vz_trans
        x_transform = self.ix_trans if meta_info['is_image'] else self.vx_trans
        z_img, z_box = z_transform(z_img, z_box)
        x_img, x_box = x_transform(x_img, x_box)

        # convert to gray-scale if necessary
        if np.random.rand() < self.gray_prob:
            z_img = img_gray(z_img)
            x_img = img_gray(x_img)

        z_img = img_norm(z_img)
        x_img = img_norm(x_img)

        num_x = x_img.size(0)
        z_img = z_img.repeat(num_x, 1, 1, 1)
        z_box = [z_box[0].clone()] * num_x

        out = {
            'z_imgs': DataContainer(z_img, stack=True, cpu_only=False),
            'x_imgs': DataContainer(x_img, stack=True, cpu_only=False),
            'z_boxes': DataContainer(z_box, stack=False, cpu_only=False),
            'x_boxes': DataContainer(x_box, stack=False, cpu_only=False),
            'flags': DataContainer(torch.tensor(x_flag, dtype=torch.uint8), stack=False, cpu_only=False)
        }

        return out

    def sample_from_dataset(self):
        """ randomly sample images from dataset_list """
        # Step 1, decide which dataset will be used
        dataset_id = np.random.choice(len(self.dataset_list), p=self.dataset_prob_list)
        _dataset = self.dataset_list[dataset_id]
        is_image = _dataset.is_image

        # Step 2, sample search region images according the the type probability distribution.
        meta_info = _dataset.sample_images()
        meta_info['is_image'] = is_image

        return meta_info

    def _load_datasets(self, dataset_cfgs):
        logger = logging.getLogger()
        self.dataset_list = []
        if isinstance(dataset_cfgs, dict):
            dataset_cfgs = [dataset_cfgs]
        assert isinstance(dataset_cfgs, list) and len(dataset_cfgs) > 0
        self.dataset_prob_list = np.ones((len(dataset_cfgs), ), dtype=np.float32)
        for ix, dataset_cfg in enumerate(dataset_cfgs):
            tic = time.time()
            _cfg = dataset_cfg.copy()
            self.dataset_prob_list[ix] = _cfg.pop('sample_weight')
            self.dataset_list.append(ImageDataset(**_cfg))
            toc = time.time()
            logger.info("Load dataset '{}' in {:.2f}s.".format(_cfg.name, (toc - tic)))

    def _load_transforms(self, transform_cfgs):
        # for image template
        self.iz_trans = transforms.build_transforms(transform_cfgs.image_z)
        # for image search regions
        self.ix_trans = transforms.build_transforms(transform_cfgs.image_x)
        # for video template
        self.vz_trans = transforms.build_transforms(transform_cfgs.video_z)
        # for video search region
        self.vx_trans = transforms.build_transforms(transform_cfgs.video_x)
