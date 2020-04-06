# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Collection of available training dataset.
Echo entry will have 4 attributes:
zip_data_path: the file path of zip container for training images.
file_data_path: the directory path of training images
ann_fn: the file path of annotations.
is_image: bool, whether the dataset is cropped from the still image set (e.g., COCO, ImageNet) or video dataset
(e.g., VID, YoutubeBB)
"""

# Required dataset entry keys
_ZIP_DATA_FN = 'zip_data'
_ANN_FN = 'ann_path'
_FILE_DATA_FN = 'file_data_dir'
_IS_IMAGE = 'is_image'


_DATASETS = dict(
    coco=dict(
        ann='coco_clean.pkl',
        storage=dict(
            zip='COCO.zip',
            lmdb='COCO_lmdb',
            file='COCO',
        ),
        is_image=True,
    ),
    got10k=dict(
        ann='got10k_train_clean.pkl',
        storage=dict(
            zip='got10k_train_images.zip',
            lmdb=None,
            file='got10k_train',
        ),
        is_image=False,
    ),
    got10k_mini=dict(
        ann='got10k_mini_clean.pkl',
        storage=dict(
            zip='got10k_mini_images.zip',
            lmdb=None,
            file='got10k_train',
        ),
        is_image=False,
    ),
    trackingnet=dict(
        ann='trackingnet_clean.pkl',
        storage=dict(
            zip='TrackingNet_images.zip',
            lmdb=None,
            file='TrackingNet_images',
        ),
        is_image=False,
    ),
    lasot=dict(
        ann='lasot_train_clean.pkl',
        storage=dict(
            zip='LaSOT_trian_images.zip',
            lmdb=None,
            file='LaSOT_trian_images',
        ),
        is_image=False,
    ),
)


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name]['ann']


def get_data_path(name, backend=None):
    backend2key = dict(
        ZipWrapper='zip',
        FileWrapper='file',
        LMDBWrapper='lmdb'
    )
    if backend not in backend2key:
        raise ValueError("Unsupported backend {}".format(backend))
    key = backend2key[backend]
    data_path = _DATASETS[name]['storage'][key]
    if data_path is None:
        raise ValueError("Unsupported backend '{}' for dataset '{}'".format(backend, name))
    return data_path


def is_image(name):
    """ Check if the dataset is image dataset """
    return _DATASETS[name]['is_image']
