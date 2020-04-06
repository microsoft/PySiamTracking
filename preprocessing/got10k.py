import argparse
import numpy as np
import os

from converter import PairAnnotationConverter


CLASSES = ['aircraft', 'ambulance', 'anteater', 'antelope', 'armadillo', 'balance car', 'ball', 'bear', 'bicycle',
           'big truck', 'bird', 'bison', 'boneshaker', 'bovine', 'bumper car', 'camel', 'canine', 'canoeing', 'car',
           'cat', 'cetacean', 'cheetah', 'chevrotain', 'crocodilian reptile', 'deer', 'dogsled', 'elephant',
           'fire engine', 'fish', 'forest goat', 'galleon', 'giraffe', 'goat', 'goat antelope', 'hagfish',
           'half track', 'handcart', 'hinny', 'hippopotamus', 'horse', 'humvee', 'hyrax', 'insectivore',
           'invertebrate', 'jaguar', 'JetLev-Flyer', 'lagomorph', 'lamprey', 'landing craft', 'larva',
           'leopard', 'lion', 'lizard', 'llama', 'luge', 'marsupial', 'mole', 'motor scooter', 'mule',
           'multistage rocket', 'musk ox', 'musteline mammal', 'natural object', 'object part', 'old world buffalo',
           'others', 'passenger ship', 'peccary', 'person', 'pickup truck', 'pinniped mammal', 'platypus', 'primate',
           'procyonid', 'pung', 'railway', 'reconnaissance vehicle', 'rhinoceros', 'road race', 'rodent',
           'rolling stock', 'sailboard', 'scooter', 'sea cow', 'self-propelled vehicle', 'sheep', 'skateboard',
           'skibob', 'sloth', 'snake', 'snow leopard', 'spacecraft', 'steamroller', 'submersible', 'swine', 'tank',
           'tapir', 'tiglon', 'toboggan', 'trailer', 'train', 'tree shrew', 'tricycle', 'troop carrier', 'tuatara',
           'turtle', 'unicycle', 'vessel', 'viverrine', 'wagon', 'warplane', 'warship', 'wheelchair', 'wild sheep',
           'zebra']


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess GOT-10k dataset')
    parser.add_argument('--nthread', dest='nthread',
                        help='number of thread',
                        default=-1,
                        type=int)
    parser.add_argument('--root_dir', dest='root_dir',
                        help='root directory of GOT-10k dataset',
                        default='',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output dir',
                        default='.',
                        type=str)
    parser.add_argument('--name', dest='name',
                        help='dataset name',
                        default='got10k_train',
                        type=str)
    parser.add_argument('--size', dest='size',
                        help='crop image size',
                        default=448,
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


def read_meta_info(file_path):
    lines = read_file_lines(file_path)
    info = dict()
    for i in range(1, len(lines)):
        p = lines[i].find(':')
        info[lines[i][:p]] = lines[i][p+1:].strip()
    if 'resolution' in info:
        res_str = info['resolution'][1:-1].replace(' ', '').split(',')
        info['resolution'] = tuple(map(int, res_str))
    return info


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    x1y1 = xywh[..., 0:2]
    wh = xywh[..., 2:4]
    x2y2 = x1y1 + (wh - 1)
    return np.concatenate((x1y1, x2y2), axis=-1)


if __name__ == '__main__':

    args = parse_args()
    root_dir = args.root_dir
    list_file = os.path.join(root_dir, 'list.txt')
    if not os.path.isdir(root_dir) or not os.path.exists(list_file):
        print("Cannot find {} or {}.\n Please check your path.".format(root_dir, list_file))
        raise FileNotFoundError

    class2id = {name: i+1 for i, name in enumerate(CLASSES)}

    cvt = PairAnnotationConverter(args.name, CLASSES)

    seq_names = read_file_lines(list_file)
    for seq_id, seq_name in enumerate(seq_names):
        gt_file = os.path.join(root_dir, seq_name, 'groundtruth.txt')
        cover_file = os.path.join(root_dir, seq_name, 'cover.label')
        absence_file = os.path.join(root_dir, seq_name, 'absence.label')
        meta_info_file = os.path.join(root_dir, seq_name, 'meta_info.ini')
        # load ground-truth
        gt_bboxes = np.array(read_file_lines(gt_file, lambda x: list(map(float, x.split(',')))), dtype=np.float32)
        gt_bboxes = xywh_to_xyxy(gt_bboxes)
        # load coverage information
        covers = np.array(read_file_lines(cover_file, int), dtype=np.int32)
        # load absence label
        absences = np.array(read_file_lines(absence_file, int), dtype=np.int32)
        # load meta info
        meta_info = read_meta_info(meta_info_file)

        frame_inds = np.arange(gt_bboxes.shape[0], dtype=np.int32)
        # remove invalid frames
        valid_inds = np.where(absences == 0)[0]
        if valid_inds.size == 0:
            print("warning: no valid annotations in {}".format(seq_name))
            continue

        frames = [os.path.join(root_dir, '{}/{:08d}.jpg'.format(seq_name, i+1)) for i in valid_inds]
        frame_inds = frame_inds[valid_inds]
        gt_bboxes = gt_bboxes[valid_inds]
        category = np.zeros((gt_bboxes.shape[0], 1), dtype=np.float32) + class2id[meta_info['major_class']]
        ignore_flag = np.zeros_like(category)
        gt_bboxes = np.concatenate((gt_bboxes, category, ignore_flag), axis=1)
        gt_bboxes = [gt_bboxes[i].reshape(1, 6) for i in range(len(gt_bboxes))]
        tracks = [[(i, 0) for i in range(len(gt_bboxes))]]

        img_size = meta_info['resolution']
        seq_info = dict(
            name=seq_name,
            frames=frames,
            bboxes=gt_bboxes,
            frame_ids=frame_inds.tolist(),
            tracks=tracks
        )

        cvt.add_video(seq_info)

        if seq_id % 200 == 0:
            print("Load annotation {}/{}".format(seq_id, len(seq_names)))

    nthread = args.nthread if args.nthread > 0 else None
    cvt.process(args.output_dir, instanc_size=args.size, num_thread=nthread)
