""" Crop image patches from COCO dataset. """

import argparse
import json
import numpy as np
import os

from preprocessing.converter import PairAnnotationConverter


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Preprocessing for COCO dataset')
    parser.add_argument('--nthread', dest='nthread',
                        help='number of thread',
                        default=-1,
                        type=int)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='the image directory path of COCO dataset',
                        default='',
                        type=str)
    parser.add_argument('--ann_path', dest='ann_path',
                        help='the annotation path',
                        default='',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output dir',
                        default='.',
                        type=str)
    parser.add_argument('--name', dest='name',
                        help='dataset name',
                        default='coco',
                        type=str)
    parser.add_argument('--size', dest='size',
                        help='crop image size',
                        default=448,
                        type=int)

    args = parser.parse_args()
    return args


def xywh_to_xyxy(xywh):
    x1y1 = xywh[..., 0:2]
    wh = xywh[..., 2:4]
    x2y2 = x1y1 + (wh - 1)
    return np.concatenate((x1y1, x2y2), axis=-1)


def build_converter(image_dir, ann_path, name='coco_train'):

    with open(ann_path, 'rb') as f:
        coco = json.load(f)

    categories = coco['categories']
    converter = PairAnnotationConverter(name=name, categories=categories)

    id2ann_index = {}
    for i, ann in enumerate(coco['annotations']):
        img_id = ann['image_id']
        if img_id not in id2ann_index:
            id2ann_index[img_id] = []
        id2ann_index[img_id].append(i)

    for i, img_info in enumerate(coco['images']):
        img_id = img_info['id']
        ann = coco['annotations'][id2ann_index[img_id]]
        img_path = os.path.join(image_dir, img_info['file_name'])

        if len(id2ann_index[img_id]) == 0:
            continue
        # collect bboxes from annotations
        xywh = np.array([coco['annotations'][k]['bbox'] for k in id2ann_index[img_id]], dtype=np.float32)
        classes = np.array([coco['annotations'][k]['category_id'] for k in id2ann_index[img_id]], dtype=np.float32)
        ignores = np.array([coco['annotations'][k]['iscrowd'] for k in id2ann_index[img_id]], dtype=np.float32)

        bboxes = xywh_to_xyxy(xywh)
        bboxes = np.hstack((bboxes, classes[:, None], ignores[:, None]))

        # keep valid annotations
        valid_inds = np.where((xywh[:, 2] > 4) & (xywh[:, 3] > 4) & (ignores < 0.5))[0]
        if len(valid_inds) == 0:
            continue

        track_list = [[(0, i)] for i in valid_inds]

        converter.add_video(
            dict(
                name=img_info['file_name'][:-4],
                frames=[img_path],
                bboxes=[bboxes],
                frame_ids=[0],
                tracks=track_list,
            )
        )

        if i % 1000 == 0:
            print('Loading annotation {}/{}'.format(i, len(coco['images'])))

    return converter


if __name__ == '__main__':

    args = parse_args()
    converter = build_converter(image_dir=args.image_dir, ann_path=args.ann_path, name=args.name)

    nthread = args.nthread if args.nthread > 0 else None
    converter.process(output_dir=args.output_dir, instanc_size=args.size, num_thread=nthread)
