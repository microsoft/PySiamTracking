import os
import numpy as np

from .base_data import Sequence, Dataset

from ..builder import BENCHMARKS
from ..utils import read_file


@BENCHMARKS.register_module
class GOT10K(Dataset):

    def __init__(self,
                 name,
                 seq_name_list,
                 data_root,
                 zip_mode=False):
        super(GOT10K, self).__init__(name=name, zip_mode=zip_mode)
        self._load_seqs(seq_name_list, data_root)

    def _load_seqs(self, seq_name_list, data_root):
        root_dir = data_root
        for seq_id, seq_name in enumerate(seq_name_list):
            file_list = os.listdir(os.path.join(root_dir, seq_name))
            img_name_list = [fn for fn in file_list if fn[-4:].lower() == '.jpg']
            frames = [os.path.join(os.path.join(root_dir, '{}/{:08d}.jpg'.format(seq_name, i + 1)))
                      for i in range(len(img_name_list))]

            gt_file = os.path.join(root_dir, seq_name, 'groundtruth.txt')
            cover_file = os.path.join(root_dir, seq_name, 'cover.label')
            absence_file = os.path.join(root_dir, seq_name, 'absence.label')

            gt_bboxes = covers = absences = None
            if os.path.exists(gt_file):
                # load ground-truth
                gt_bboxes = np.array(read_file(gt_file, lambda x: list(map(float, x.split(',')))),
                                     dtype=np.float32)
            if os.path.exists(cover_file):
                # load coverage information
                covers = np.array(read_file(cover_file, int), dtype=np.int32)
            if os.path.exists(absence_file):
                # load absence label
                absences = np.array(read_file(absence_file, int), dtype=np.int32)

            self._seqs.append(Sequence(name=seq_name, frames=frames, gt_rects=gt_bboxes,
                                       attrs=None, cover=covers, absence=absences))


@BENCHMARKS.register_module
class GOT10KVal(GOT10K):

    def __init__(self,
                 data_root='data/benchmark/',
                 zip_mode=False):
        filename = os.path.join(os.path.dirname(__file__), 'meta_info', 'got10k_val_list.txt')
        seq_name_list = read_file(filename)
        super(GOT10KVal, self).__init__(name='got10k_val',
                                        seq_name_list=seq_name_list,
                                        data_root=os.path.join(data_root, 'got10k', 'val'),
                                        zip_mode=zip_mode)


@BENCHMARKS.register_module
class GOT10KTest(GOT10K):

    def __init__(self,
                 data_root='data/benchmark/',
                 zip_mode=False):
        filename = os.path.join(os.path.dirname(__file__), 'meta_info', 'got10k_test_list.txt')
        seq_name_list = read_file(filename)
        super(GOT10KTest, self).__init__(name='got10k_test',
                                         seq_name_list=seq_name_list,
                                         data_root=os.path.join(data_root, 'got10k', 'test'),
                                         zip_mode=zip_mode)

