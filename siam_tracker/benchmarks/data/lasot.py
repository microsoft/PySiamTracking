import os
import json
import numpy as np

from .base_data import Dataset, Sequence

from ..builder import BENCHMARKS


@BENCHMARKS.register_module
class LaSOT(Dataset):

    def __init__(self,
                 name='lasot',
                 meta_path='',
                 zip_mode=False,
                 data_root='data/benchmark/'):
        super(LaSOT, self).__init__(name=name, zip_mode=zip_mode)
        self._lasot_root = os.path.join(data_root, 'LaSOT')
        self._load_seqs(meta_path)

    def _load_seqs(self, meta_path: str):
        with open(meta_path, 'r') as f:
            seq_info_list = json.load(f)

        for seq_info in seq_info_list:
            name = seq_info['name']
            seq_dir = os.path.join(self._lasot_root, name.split('-')[0], name)
            gt_path = os.path.join(seq_dir, 'groundtruth.txt')
            occ_path = os.path.join(seq_dir, 'full_occlusion.txt')
            oov_path = os.path.join(seq_dir, 'out_of_view.txt')
            gt_boxes = np.genfromtxt(gt_path, dtype=np.float64, delimiter=',')
            _occ_info = np.genfromtxt(occ_path, dtype=np.int, delimiter=',').reshape(-1)
            _oov_info = np.genfromtxt(oov_path, dtype=np.int, delimiter=',').reshape(-1)
            absence = np.greater(_occ_info + _oov_info, 0)
            assert len(absence) == len(gt_boxes)
            frame_fmt = os.path.join(seq_dir, 'img', '{:08d}.jpg')
            frames = [frame_fmt.format(i + 1) for i in range(len(gt_boxes))]
            if self.zip_mode:
                zip_path = os.path.join(self._lasot_root, 'zips', '{}.zip'.format(name))
            else:
                zip_path = None
            self._seqs.append(Sequence(name=name, frames=frames, gt_rects=gt_boxes,
                                       attrs=seq_info['attrs'], zip_path=zip_path, absence=absence))


@BENCHMARKS.register_module
class LaSOTTest(LaSOT):

    def __init__(self,
                 zip_mode=False,
                 data_root='data/benchmark/'):
        meta_path = os.path.join(os.path.dirname(__file__), 'meta_info/lasot_test.json')
        super(LaSOTTest, self).__init__(name='lasottest',
                                        meta_path=meta_path,
                                        zip_mode=zip_mode,
                                        data_root=data_root)


@BENCHMARKS.register_module
class LaSOTFull(LaSOT):

    def __init__(self,
                 zip_mode=False,
                 data_root='data/benchmark/'):
        meta_path = os.path.join(os.path.dirname(__file__), 'meta_info/lasot_full.json')
        super(LaSOTFull, self).__init__(name='lasotfull',
                                        meta_path=meta_path,
                                        zip_mode=zip_mode,
                                        data_root=data_root)
