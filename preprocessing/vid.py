import _init_paths
import numpy as np
import argparse
import glob
import xml.etree.ElementTree as ET

from os.path import join, basename, exists
from os import listdir

from preprocessing.converter import PairAnnotationConverter

CLASS_NAMES = [
    'n02691156', 'n02958343', 'n02834778', 'n02084071', 'n02411705', 'n02402425', 'n03790512', 'n02509815',
    'n04530566', 'n02121808', 'n01503061', 'n02510455', 'n02419796', 'n04468005', 'n02062744', 'n02374451',
    'n01662784', 'n01726692', 'n02129165', 'n02131653', 'n02484322', 'n02355227', 'n02118333', 'n02342885',
    'n02391049', 'n02129604', 'n01674464', 'n02503517', 'n02324045', 'n02924116'
]


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess for VID dataset')
    parser.add_argument('--nthread', dest='nthread',
                        help='number of thread',
                        default=-1,
                        type=int)
    parser.add_argument('--root_dir', dest='root_dir',
                        help='the root directory path of VID dataset',
                        default='',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output dir',
                        default='.',
                        type=str)
    parser.add_argument('--name', dest='name',
                        help='dataset name',
                        default='vid',
                        type=str)
    parser.add_argument('--size', dest='size',
                        help='crop image size',
                        default=448,
                        type=int)

    args = parser.parse_args()
    return args


def build_converter(name, vid_root_dir):
    class2id = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}

    ann_root_dir = join(vid_root_dir, 'Annotations', 'VID')
    img_root_dir = join(vid_root_dir, 'Data', 'VID')

    sub_sets = [
        'train/ILSVRC2015_VID_train_0000',
        'train/ILSVRC2015_VID_train_0001',
        'train/ILSVRC2015_VID_train_0002',
        'train/ILSVRC2015_VID_train_0003',
        'val/',
    ]

    converter = PairAnnotationConverter(name=name, categories=CLASS_NAMES)
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_root_dir, sub_set)
        video_names = sorted(listdir(sub_set_base_path))
        for vi, video_name in enumerate(video_names):
            video = dict(
                name=video_name,
                frames=[],
                bboxes=[],
                frame_ids=[],
                tracks=[],
            )

            video_base_path = join(sub_set_base_path, video_name)
            xmls = sorted(glob.glob(join(video_base_path, '*.xml')))

            id_set = {}
            track_list = []

            for frame_id, xml in enumerate(xmls):

                # skip some frames to save storage space.
                if frame_id % 2:
                    continue

                xmltree = ET.parse(xml)
                objects = xmltree.findall('object')
                objs = []
                for obj_id, object_iter in enumerate(objects):
                    cls_id = class2id[(object_iter.find('name')).text]
                    bndbox = object_iter.find('bndbox')

                    objs.append([
                        int(bndbox.find('xmin').text),
                        int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text),
                        int(bndbox.find('ymax').text),
                        cls_id,
                        0
                    ])

                    trackid = int(object_iter.find('trackid').text)
                    if trackid not in id_set:
                        id_set[trackid] = len(track_list)
                        track_list.append([])
                    track_list[id_set[trackid]].append((frame_id, obj_id))

                video['bboxes'].append(np.array(objs, dtype=np.float32))
                img_path = join(img_root_dir, sub_set, video_name, basename(xml).replace('xml', 'JPEG'))
                assert exists(img_path), "Cannot find '{}'".format(img_path)
                video['frames'].append(img_path)
                video['frame_ids'].append(frame_id)
                video['tracks'] = track_list

            if vi % 100 == 0:
                print("[{}] Load annotations {}/{}".format(sub_set, vi, len(video_names)))
            converter.add_video(video)

    return converter


if __name__ == '__main__':
    args = parse_args()
    converter = build_converter(name=args.name, vid_root_dir=args.root_dir)

    nthread = args.nthread if args.nthread > 0 else None
    converter.process(output_dir=args.output_dir, instanc_size=args.size, num_thread=nthread)
