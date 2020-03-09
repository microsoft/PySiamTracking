# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import numpy as np
from collections import OrderedDict
import cv2
import _pickle as pickle

from . import data_catelog
from .storage import build_storage


class ImageDataset(object):

    def __init__(self,
                 name,
                 data_root,
                 storage=None,
                 max_frame_dist=1000,
                 sample_type_prob=[(0.7, 0.2, 0.1)],
                 max_category_num=15000):
        """ ImageDataset is a backend for loading the training images from a dataset.

        Args:
            dataset_name (string): the name of dataset.
            max_frame_dist (int): the maximum frame distance when sampling image pairs.
            max_category_num (int): since some categories dominate the dataset (e.g. 'person' in COCO), if we
                              uniformly sample images, some uncommon categories might be difficult to be selected.
                              Thus we set a limitation on the number of instances in echo category.
        """
        self.dataset_name = name
        self.is_image = data_catelog.is_image(name)
        self.max_category_num = max_category_num
        self.max_frame_dist = max_frame_dist
        self.storage = storage
        self.data_root = data_root
        self.sample_type_prob = sample_type_prob if isinstance(sample_type_prob, list) else [sample_type_prob]

        self.category_id_list = []
        self.seqs = []
        self.category_id2name = OrderedDict()
        self.category_id2anno = OrderedDict()

        self._load_dataset()

    def num_categories(self):
        """ Return number of categories """
        return len(self.category_id_list)

    def prepare_output(self, id_info_list):
        """ Generate output dict by id_info_list. """

        box_list = []
        img_list = []
        same_entity_flag_list = []
        core_seq_id = id_info_list[0][0]

        for id_info in id_info_list:
            seq_id, img_id = id_info
            seq_info = self.seqs[seq_id]

            # load image pixel values
            imgbuf = self.storage[seq_info['frames'][img_id]]
            img = cv2.imdecode(
                np.fromstring(imgbuf, dtype=np.uint8), cv2.IMREAD_COLOR)

            img_list.append(img)
            box_list.append(seq_info['bboxes'][img_id].copy())
            same_entity_flag_list.append(core_seq_id == seq_id)

        out = dict(
            img_list=img_list,
            box_list=box_list,
            flag_list=same_entity_flag_list
        )

        return out

    def random_sample_by_category_id(self, cat_id):
        """ Randomly sample objects by category id"""
        num_annos = self.category_id2anno[cat_id].shape[0]
        _anno_id = np.random.choice(num_annos)
        seq_id, img_id = self.category_id2anno[cat_id][_anno_id, :]
        return seq_id, img_id

    def sample_images(self):
        # random sample a category
        core_cat_id = self.category_id_list[np.random.choice(self.num_categories(), p=self.sample_category_prob)]
        core_id = self.random_sample_by_category_id(core_cat_id)
        id_list = [core_id]
        for _type_prob in self.sample_type_prob:
            image_type = np.random.choice(3, p=_type_prob)
            if image_type == 0:
                id_list.append(self._same_entity_as_z(core_id))
            elif image_type == 1:
                id_list.append(self.random_sample_by_category_id(core_cat_id))
            else:
                cat_id = self.category_id_list[np.random.choice(self.num_categories(), p=self.sample_category_prob)]
                id_list.append(self.random_sample_by_category_id(cat_id))
        return self.prepare_output(id_list)

    def _same_entity_as_z(self, z_id):
        x_seq_id = z_id[0]
        if self.is_image:
            x_img_id = z_id[1]
        else:
            frame_inds = self.seqs[x_seq_id]['frame_id']
            valid_inds = np.where(np.abs(frame_inds - frame_inds[z_id[1]]) <= self.max_frame_dist)[0]
            x_img_id = np.random.choice(valid_inds)
        x_id = (x_seq_id, x_img_id)
        return x_id

    def _load_json_annotation(self, anno_file_path):
        """ Json annotation format:
        {
            'categories': [
                {
                    'id': 1 (int)
                    'name': 'Person' (string)
                },
                ...
            ],
            'seqs': [
                [
                    {
                        'bbox'：[[x1, y1, x2, y2, category_id, is_ignore]],
                        'file_name': 'coco/1.jpg' (string)
                        'frame_id': 0 (int)
                    },  (per frame)
                    ...
                ], (per video)
                ...
            ]
        }
        """
        raise NotImplementedError

    def _load_pkl_annotation(self, anno_file_path):
        with open(anno_file_path, 'rb') as f:
            annos = pickle.load(f)

        # update categories
        for cat_info in annos['categories']:
            cat_id, cat_name = int(cat_info['id']), cat_info['name']
            assert cat_id >= 0, "The category index should be larger than 0. (Got {})".format(cat_id)
            if cat_id not in self.category_id2name:
                self.category_id_list.append(cat_id)
                self.category_id2name[cat_id] = cat_name
            else:
                raise ValueError("Duplicated category index. "
                                 "({} {} {})".format(cat_id, cat_name, self.category_id2name[cat_id]))

        # update video information
        for seq_id, seq_info in enumerate(annos['seqs']):
            assert len(seq_info['bboxes']) == len(seq_info['frame_id']) == len(seq_info['frames'])
            # clean and filter some invalid boxes
            for frame_id in range(len(seq_info['frames'])):
                bboxes = seq_info['bboxes'][frame_id]
                same_cat_inds = np.where(bboxes[:, 4] == bboxes[0, 4])[0]
                bboxes = bboxes[same_cat_inds]
                # some invalid boxes, whose width or height is less than 0, will be removed.
                valid_inds = np.logical_and((bboxes[:, 2] > bboxes[:, 0] + 1),
                                            (bboxes[:, 3] > bboxes[:, 1] + 1))
                if not np.all(valid_inds):
                    assert valid_inds[0]
                    bboxes = bboxes[valid_inds]
                seq_info['bboxes'][frame_id] = bboxes
            seq_info['frame_id'] = np.array(seq_info['frame_id'], dtype=np.float32)
        self.seqs = annos['seqs']

    def _load_dataset(self):
        """ Initialization the dataset """
        data_path = os.path.join(self.data_root, data_catelog.get_data_path(self.dataset_name, self.storage['type']))
        self.storage = build_storage(self.storage, data_path)

        # load annotations
        anno_file_path = os.path.join(self.data_root, data_catelog.get_ann_fn(self.dataset_name))
        if anno_file_path.endswith('.pkl'):
            # use pickle annotation loader
            self._load_pkl_annotation(anno_file_path)
        elif anno_file_path.endswith('.json'):
            # use json annotations loader
            self._load_json_annotation(anno_file_path)
        else:
            raise ValueError("Unsupport annotation type {}".format(anno_file_path))

        self._collect_available_images()

    def _collect_available_images(self):
        """ re-organize the dataset for fast indexing"""

        # counts in different categories
        category_count = {cat_id: 0 for cat_id in self.category_id2name.keys()}
        for seq_id, seq_info in enumerate(self.seqs):
            cat_id = int(seq_info['category'])
            category_count[cat_id] += len(seq_info['bboxes'])

        # given a category id, return a list including all frame indexes which has target category
        self.category_id2anno = OrderedDict()
        for cat_id in self.category_id_list:
            # 2 dimensions mean sequence-level index & frame-level index
            self.category_id2anno[cat_id] = np.zeros((category_count[cat_id], 2), dtype=np.int32)

        current_count = {k: 0 for k in category_count.keys()}
        for seq_id, seq_info in enumerate(self.seqs):
            cat_id = int(seq_info['category'])
            for k in range(len(seq_info['bboxes'])):
                self.category_id2anno[cat_id][current_count[cat_id], :] = (seq_id, k)
                current_count[cat_id] += 1

        # calculate the sample probability over difference categories
        self.sample_category_prob = np.zeros((self.num_categories(),), np.float32)
        for ix, cat_id in enumerate(self.category_id_list):
            self.sample_category_prob[ix] = category_count[cat_id]
        self.sample_category_prob = np.clip(self.sample_category_prob, a_min=0, a_max=self.max_category_num)
        self.sample_category_prob /= self.sample_category_prob.sum()
