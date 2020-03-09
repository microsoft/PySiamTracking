import os
import numpy as np
import zipfile

from .base_data import Dataset, Sequence
from ..builder import BENCHMARKS
from ..utils import read_file


@BENCHMARKS.register_module
class TrackingNet(Dataset):

    zero_based_index = False

    def __init__(self,
                 zip_mode=False,
                 data_root='data/benchmark'):
        super(TrackingNet, self).__init__('trackingnet', zip_mode=zip_mode)
        self._trackingnet_root = os.path.join(data_root, 'TrackingNet', 'TEST')

    def _load_seqs(self):
        file_path = os.path.join(os.path.dirname(__file__), 'meta_info/trackingnet_test_list.txt')
        seq_name_list = read_file(file_path)
        for name in seq_name_list:
            gt_path = os.path.join(self._trackingnet_root, 'anno', '{}.txt'.format(name))
            gt_boxes = np.genfromtxt(gt_path, dtype=np.float64, delimiter=',').reshape(1, 4)
            img_dir = os.path.join(self._trackingnet_root, 'frames', name)
            if os.path.isdir(img_dir):
                img_list = [fn for fn in os.listdir(img_dir) if fn[-4:] == '.jpg']
            else:
                zip_path = os.path.join(self._trackingnet_root, 'zips', '{}.zip'.format(name))
                assert os.path.exists(zip_path)
                zip_file = zipfile.ZipFile(zip_path)
                img_list = []
                for fn in zip_file.filelist:
                    if fn.filename.endswith('.jpg'):
                        img_list.append(fn)

            frame_fmt = os.path.join(img_dir, '{}.jpg')
            frames = [frame_fmt.format(i) for i in range(len(img_list))]
            if self.zip_mode:
                zip_path = os.path.join(self._trackingnet_root, 'zips', '{}.zip'.format(name))
            else:
                zip_path = None
            self._seqs.append(Sequence(name=name, frames=frames, gt_rects=gt_boxes, zip_path=zip_path))
