import os
import numpy as np

from .base_data import Dataset, Sequence
from ..utils import read_file
from ..builder import BENCHMARKS


@BENCHMARKS.register_module
class VOT(Dataset):

    def __init__(self,
                 name='vot',
                 vot_root='data/benchmark/vot17/',
                 zip_mode=False):
        super(VOT, self).__init__(name=name, zip_mode=zip_mode)
        self._load_seqs(vot_root)

    def select_tag(self, video_name: str, tag: str, start: int = 0, end: int = 0):
        seq = self.seqs[self.get(video_name)]
        if tag == 'all':
            all_tags = [1] * len(seq)
            return all_tags[start:end]
        elif hasattr(seq, tag):
            return getattr(seq, tag)[start:end]
        else:
            raise NotImplementedError("Cannot find tag '{}'".format(tag))

    def _load_seqs(self, vot_root: str) -> None:
        list_file = os.path.join(vot_root, 'list.txt')
        seq_name_list = read_file(list_file)
        for seq_name in seq_name_list:
            seq = self._load_single_seq(os.path.join(vot_root, seq_name))
            self._seqs.append(seq)

    def _load_single_seq(self, data_dir: str) -> Sequence:
        name = os.path.basename(data_dir)
        # load ground-truth annotation
        gt_rects = np.loadtxt(os.path.join(data_dir, 'groundtruth.txt'), dtype=np.float64, delimiter=',')
        # load frames
        img_dir = os.path.join(data_dir, 'color')
        frames = [os.path.join(img_dir, '{:08d}.jpg'.format(i+1)) for i in range(len(gt_rects))]

        kwargs = dict()
        for att in ['occlusion', 'illum_change', 'motion_change', 'size_change', 'camera_motion']:
            tag_file = os.path.join(data_dir, '{}.tag'.format(att))
            if os.path.exists(tag_file):
                att_tags = np.loadtxt(tag_file).astype(np.bool)
                if len(att_tags) < len(frames):
                    _pad = np.zeros((len(frames),), dtype=np.bool)
                    _pad[:len(att_tags)] = att_tags
                    att_tags = _pad
                kwargs[att] = att_tags

        if self.zip_mode:
            seq_name = os.path.basename(data_dir)
            zip_path = os.path.join(data_dir, '..', 'zips', '{}.zip'.format(seq_name))
        else:
            zip_path = None
        return Sequence(name, frames, gt_rects, attrs=None, zip_path=zip_path, **kwargs)


@BENCHMARKS.register_module
class VOT17(VOT):

    def __init__(self, data_root='data/benchmark/', zip_mode=False):
        super(VOT17, self).__init__(name='vot17',
                                    vot_root=os.path.join(data_root, 'vot17'),
                                    zip_mode=zip_mode)


@BENCHMARKS.register_module
class VOT16(VOT):

    def __init__(self, data_root='data/benchmark/', zip_mode=False):
        super(VOT16, self).__init__(name='vot16',
                                    vot_root=os.path.join(data_root, 'vot16'),
                                    zip_mode=zip_mode)
