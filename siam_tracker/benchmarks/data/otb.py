""" Object Tracking Benchmark
http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
Online object tracking: A benchmark. (CVPR 2013)
Wu, Yi, Jongwoo Lim, and Ming-Hsuan Yang.
"""
import os
import json
import numpy as np

from .base_data import Dataset, Sequence
from ..builder import BENCHMARKS


@BENCHMARKS.register_module
class OTB(Dataset):

    zero_based_index = False

    def __init__(self,
                 split,
                 name='otb',
                 zip_mode=False,
                 data_root='data/benchmark/'):
        super(OTB, self).__init__(name, zip_mode)
        if isinstance(split, str):
            split = split.split(',')
        assert isinstance(split, list)
        self._otb_root = os.path.join(data_root, 'otb')
        self._load_seqs(split)

    def _load_seqs(self, seq_name_list):
        otb_info_path = os.path.join(os.path.dirname(__file__), 'meta_info', 'otb.json')
        with open(otb_info_path, 'r') as f:
            all_seq_info_list = json.load(f)

        name2id = {seq_info['name']: ix for ix, seq_info in enumerate(all_seq_info_list)}
        seq_info_list = [all_seq_info_list[name2id[seq_name]] for seq_name in seq_name_list]
        for seq_name, seq_info in zip(seq_name_list, seq_info_list):
            anno_path = os.path.join(self._otb_root, seq_info['anno_path'])
            gt_rects = self._load_annotaion_single(anno_path)
            if seq_name == 'Tiger1':
                gt_rects = gt_rects[5:]  # skip the first 5 frames.

            frame_fm = os.path.join(self._otb_root, seq_info['path'], '{idx:0{nz}}.{ext}')
            frames = []
            for idx in range(seq_info['startFrame'], seq_info['endFrame'] + 1):
                frames.append(frame_fm.format(idx=idx, nz=seq_info['nz'], ext=seq_info['ext']))
            frames = frames[:len(gt_rects)]  # same size with ground-truth annotations

            omits = np.zeros((len(frames), ), dtype=np.bool)
            for omit in seq_info['omit']:
                omits[omit[0] - seq_info['startFrame']:omit[1] - seq_info['startFrame']] = True

            if self.zip_mode:
                zip_path = os.path.join(self._otb_root, 'zips/{}.zip'.format(seq_name))
            else:
                zip_path = None

            self._seqs.append(Sequence(name=seq_name, frames=frames, gt_rects=gt_rects,
                                       attrs=seq_info['attrs'], zip_path=zip_path, omits=omits))

    @staticmethod
    def _load_annotaion_single(anno_path):
        assert os.path.isfile(anno_path), "Cannot find annotation file: {}".format(os.path.abspath(anno_path))
        with open(anno_path, 'r') as f:
            lines = f.read().splitlines()
        gt_rects = []
        for line in lines:
            if '\t' in line:
                gt_rects.append(list(map(int, line.strip().split('\t'))))
            elif ',' in line:
                gt_rects.append(list(map(int, line.strip().split(','))))
            elif ' ' in line:
                gt_rects.append(list(map(int, line.strip().split(' '))))
        gt_rects = np.array(gt_rects, dtype=np.float64)

        # For some sequences in OTB, the last box annotation is invalid. For fair comparison, we manually remove
        # these annotations.
        invalid_count = 0
        for i in range(1, len(gt_rects)):
            if np.all(np.array(gt_rects[-i]) <= 0):
                invalid_count += 1
                break
        if invalid_count > 0:
            gt_rects = gt_rects[:-invalid_count]

        return gt_rects


@BENCHMARKS.register_module
class OTB50(OTB):
    def __init__(self, *args, **kwargs):
        super(OTB50, self).__init__(split=_get_tb50_list(),
                                    name='otb50',
                                    *args, **kwargs)


@BENCHMARKS.register_module
class OTB100(OTB):
    def __init__(self, *args, **kwargs):
        super(OTB100, self).__init__(split=_get_tb100_list(),
                                     name='otb100',
                                     *args, **kwargs)


@BENCHMARKS.register_module
class OTB2013(OTB):
    def __init__(self, *args, **kwargs):
        super(OTB2013, self).__init__(split=_get_cvpr13_list(),
                                      name='otb2013',
                                      *args, **kwargs)


def _get_cvpr13_list():
    return ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', 'Coke', 'Couple', 'Crossing', 'David',
            'David2', 'David3', 'Deer', 'Dog1', 'Doll', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace',
            'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Ironman', 'Jogging_1', 'Jogging_2',
            'Jumping', 'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer1',
            'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Trellis',
            'Walking', 'Walking2', 'Woman']


def _get_tb50_list():
    return ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1',
            'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds', 'David', 'Deer', 'Diving', 'DragonBaby',
            'Dudek', 'Football', 'Freeman4', 'Girl', 'Human3', 'Human4', 'Human6', 'Human9', 'Ironman', 'Jump',
            'Jumping', 'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam', 'Shaking', 'Singer2', 'Skating1',
            'Skating2_1', 'Skating2_2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis', 'Walking',
            'Walking2', 'Woman']


def _get_tb100_list():
    return ['Basketball', 'Biker', 'Bird1', 'Bird2', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4',
            'BlurFace', 'BlurOwl', 'Board', 'Bolt', 'Bolt2', 'Box', 'Boy', 'Car1', 'Car2', 'Car24', 'Car4', 'CarDark',
            'CarScale', 'ClifBar', 'Coke', 'Couple', 'Coupon', 'Crossing', 'Crowds', 'Dancer', 'Dancer2', 'David',
            'David2', 'David3', 'Deer', 'Diving', 'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2',
            'Fish', 'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2',
            'Gym', 'Human2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman',
            'Jogging_1', 'Jogging_2', 'Jump', 'Jumping', 'KiteSurf', 'Lemming', 'Liquor', 'Man', 'Matrix', 'Mhyang',
            'MotorRolling', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Singer2', 'Skater',
            'Skater2', 'Skating1', 'Skating2_1', 'Skating2_2', 'Skiing', 'Soccer', 'Subway', 'Surfer', 'Suv',
            'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trans', 'Trellis', 'Twinnings', 'Vase', 'Walking', 'Walking2',
            'Woman']
