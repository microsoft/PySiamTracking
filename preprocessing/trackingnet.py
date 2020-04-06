import os
import numpy as np
import argparse
import cv2

from converter import PairAnnotationConverter

YTBB_CLASSES = ['person', 'bird', 'bicycle', 'boat', 'bus',
                    'bear', 'cow', 'cat', 'giraffe', 'potted plant',
                    'horse', 'motorcycle', 'knife', 'airplane', 'skateboard',
                    'train', 'truck', 'zebra', 'toilet', 'dog',
                    'elephant', 'umbrella', 'none', 'car']


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess for TrackingNet dataset')
    parser.add_argument('--nthread', dest='nthread',
                        help='number of thread',
                        default=-1,
                        type=int)
    parser.add_argument('--root_dir', dest='root_dir',
                        help='the root directory path of LaSOT dataset',
                        default='',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output dir',
                        default='.',
                        type=str)
    parser.add_argument('--name', dest='name',
                        help='dataset name',
                        default='lasot_train',
                        type=str)
    parser.add_argument('--size', dest='size',
                        help='crop image size',
                        default=448,
                        type=int)
    parser.add_argument('--downsample', dest='downsample',
                        help='sample one per N frame.',
                        default=10,
                        type=int)
    args = parser.parse_args()
    return args


def read_file_lines(file_path, process_func=None):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    lines = [line for line in lines if line.strip() != '']  # remove null
    if process_func is not None:
        lines = [process_func(line) for line in lines]
    return lines


def load_dataset(root_dir):
    video_list = []

    for sp in range(12):
        sub_dir = os.path.join(root_dir, 'TRAIN_{}'.format(sp))

        class_lines = read_file_lines(os.path.join(root_dir, 'TRAIN_{}_classinfo.txt'.format(sp)))
        name2cat_id = dict()
        for line in class_lines:
            name, cat_id = line.split(',')
            cat_id = int(cat_id)
            if cat_id == -1:
                cat_id = 23
            cat_id += 1
            name2cat_id[name] = cat_id

        frame_dir = os.path.join(sub_dir, 'frames')
        seq_name_list = [fn for fn in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, fn))]
        for seq_name in seq_name_list:
            img_dir = os.path.join(frame_dir, seq_name)
            anno_path = os.path.join(sub_dir, 'anno', '{}.txt'.format(seq_name))
            assert os.path.exists(anno_path)
            video_list.append((seq_name, name2cat_id[seq_name], img_dir, anno_path))

    print("All {} videos".format(len(video_list)))
    return video_list


def xywh_to_xyxy(xywh):
    x1y1 = xywh[..., 0:2]
    wh = xywh[..., 2:4]
    x2y2 = x1y1 + wh
    return np.concatenate((x1y1, x2y2), axis=-1)


def build_converter(root_dir, downsample, min_box_size=10):

    cvt = PairAnnotationConverter(name='trackingnet', categories=YTBB_CLASSES)
    video_list = load_dataset(root_dir)
    for idx, (seq_name, cat_id, img_dir, anno_path) in enumerate(video_list):
        # load ground-truth
        gt_bboxes = np.array(read_file_lines(anno_path, lambda x: lambda x: list(map(float, x.split(',')))),
                             dtype=np.float32)
        gt_bboxes = xywh_to_xyxy(gt_bboxes)
        num_frames = len(gt_bboxes)
        assert gt_bboxes.shape[1] == 4
        frames = [os.path.join(img_dir, '{}.jpg'.format(i)) for i in range(len(gt_bboxes))]

        # check file exits
        img_shape = None
        for frame_path in frames:
            assert os.path.exists(frame_path)
            if img_shape is None:
                img_shape = cv2.imread(frame_path).shape[0:2]

        # Reset ground-truth boxes
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[0])

        wh = gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2]
        valid_size = wh > 10

        valid = np.zeros((len(gt_bboxes, )), np.bool)
        valid[0::downsample] = True
        valid[~valid_size] = False

        valid_inds = np.where(valid)[0]

        frames = [frames[i] for i in valid_inds]
        gt_bboxes = np.concatenate((gt_bboxes,
                                    np.full(shape=(len(gt_bboxes), 1), value=cat_id),
                                    np.zeros((len(gt_bboxes, 1)))), axis=1)  # [N, 6]
        gt_bboxes = [gt_bboxes[i].reshape(1, 6) for i in valid_inds]
        frame_ids = valid_inds
        tracks = [[(i, 0) for i in range(len(valid_inds))]]

        print("[{}] {} --> {} ({}/{})".format(seq_name, num_frames, len(valid_inds), idx, len(video_list)))
        if len(valid_inds) >= 2:
            cvt.add_video(
                dict(
                    name=seq_name,
                    frames=frames,
                    bboxes=gt_bboxes,
                    frame_ids=frame_ids,
                    tracks=tracks
                )
            )
    return cvt


if __name__ == '__main__':
    args = parse_args()
    cvt = build_converter(args.root_dir, args.downsample)
    nthread = args.nthread if args.nthread > 0 else None
    cvt.process(args.output_dir, instanc_size=args.size, num_thread=nthread)
