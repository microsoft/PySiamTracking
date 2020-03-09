import os
import numpy as np
import argparse

from preprocessing.converter import PairAnnotationConverter


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess for LaSOT-train dataset')
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
                        default=3,
                        type=int)
    args = parser.parse_args()
    return args


def load_training_set(root_dir):
    testset_names = get_testset()
    assert len(testset_names) == 280
    lasot_classes = get_classes()
    dir_list = []
    for class_name in lasot_classes:
        class_dir = os.path.join(root_dir, class_name)
        assert os.path.isdir(class_dir), 'Cannot find {}'.format(class_dir)
        name_list = [fn for fn in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, fn))]
        name_list = [fn for fn in name_list if fn not in testset_names]  # exclude test set
        dir_list.extend(['{}/{}'.format(class_name, fn) for fn in name_list])
    assert len(dir_list) == 1400 - 280
    return dir_list


def read_file_lines(file_path, process_func=None):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    lines = [line for line in lines if line.strip() != '']  # remove null
    if process_func is not None:
        lines = [process_func(line) for line in lines]
    return lines


def build_converter(root_dir, downsample, min_box_size=10):
    training_dir_list = load_training_set(root_dir)
    cvt = PairAnnotationConverter(name='lasot_train', categories=get_classes())

    name2id = {c['name']:c['id'] for c in cvt.categories}
    for idx, training_dir in enumerate(training_dir_list):
        seq_name = training_dir.split('/')[-1]
        seq_class = training_dir.split('/')[0]
        seq_class_id = name2id[seq_class]
        img_dir = os.path.join(root_dir, training_dir, 'img')
        gt_file = os.path.join(root_dir, training_dir, 'groundtruth.txt')
        occ_file = os.path.join(root_dir, training_dir, 'full_occlusion.txt')
        ofv_file = os.path.join(root_dir, training_dir, 'out_of_view.txt')
        # load ground-truth file
        gt_list = read_file_lines(gt_file, lambda x: list(map(float, x.split(','))))
        gt_bboxes = []
        for gt in gt_list:
            x1, y1, w, h = gt
            gt_bboxes.append(np.array([x1, y1, x1 + w, y1 + h, seq_class_id, 0], dtype=np.float32).reshape(1, 6))

        num_frames = len(gt_bboxes)
        # check file exists
        frames = [os.path.join(img_dir, '{:08d}.jpg'.format(i+1)) for i in range(num_frames)]
        for frame_path in frames:
            assert os.path.exists(frame_path), frame_path

        frame_ids = [i+1 for i in range(num_frames)]

        valid_inds = np.zeros((num_frames, ), dtype=np.bool)
        valid_inds[0::downsample] = True
        occlusion = np.array(read_file_lines(occ_file, lambda x: list(map(int, x.split(',')))), dtype=np.int32)[0]
        valid_inds[occlusion > 0] = False
        out_of_view = np.array(read_file_lines(ofv_file, lambda x: list(map(int, x.split(',')))), dtype=np.int32)[0]
        valid_inds[out_of_view > 0] = False
        for i in range(len(gt_bboxes)):
            box = gt_bboxes[i][0]
            if box[2] - box[0] < min_box_size or box[3] - box[1] < min_box_size:
                valid_inds[i] = False

        valid_inds = np.where(valid_inds)[0]
        print("[{}] {} --> {} ({}/{})".format(seq_name, num_frames, len(valid_inds), idx, len(training_dir_list)))
        if len(valid_inds) >= 2:
            tracks = [[(i, 0) for i in range(len(valid_inds))]]
            frame_ids = [frame_ids[i] for i in valid_inds]
            frames = [frames[i] for i in valid_inds]
            gt_bboxes = [gt_bboxes[i] for i in valid_inds]
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


def get_classes():
    classes = [
        'airplane', 'basketball', 'bear', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bus', 'car', 'cat', 'cattle',
        'chameleon', 'coin', 'crab', 'crocodile', 'cup', 'deer', 'dog', 'drone', 'electricfan', 'elephant', 'flag',
        'fox', 'frog', 'gametarget', 'gecko', 'giraffe', 'goldfish', 'gorilla', 'guitar', 'hand', 'hat', 'helmet',
        'hippo', 'horse', 'kangaroo', 'kite', 'leopard', 'licenseplate', 'lion', 'lizard', 'microphone', 'monkey',
        'motorcycle', 'mouse', 'person', 'pig', 'pool', 'rabbit', 'racing', 'robot', 'rubicCube', 'sepia', 'shark',
        'sheep', 'skateboard', 'spider', 'squirrel', 'surfboard', 'swing', 'tank', 'tiger', 'train', 'truck', 'turtle',
        'umbrella', 'volleyball', 'yoyo', 'zebra'
    ]
    return classes


def get_testset():
    lasot_test = [
        "airplane-1", "airplane-9", "airplane-13", "airplane-15", "basketball-1", "basketball-6", "basketball-7",
        "basketball-11", "bear-2", "bear-4", "bear-6", "bear-17", "bicycle-2", "bicycle-7", "bicycle-9", "bicycle-18",
        "bird-2", "bird-3", "bird-15", "bird-17", "boat-3", "boat-4", "boat-12", "boat-17", "book-3", "book-10",
        "book-11",
        "book-19", "bottle-1", "bottle-12", "bottle-14", "bottle-18", "bus-2", "bus-5", "bus-17", "bus-19", "car-2",
        "car-6", "car-9", "car-17", "cat-1", "cat-3", "cat-18", "cat-20", "cattle-2", "cattle-7", "cattle-12",
        "cattle-13",
        "spider-14", "spider-16", "spider-18", "spider-20", "coin-3", "coin-6", "coin-7", "coin-18", "crab-3", "crab-6",
        "crab-12", "crab-18", "surfboard-12", "surfboard-4", "surfboard-5", "surfboard-8", "cup-1", "cup-4", "cup-7",
        "cup-17", "deer-4", "deer-8", "deer-10", "deer-14", "dog-1", "dog-7", "dog-15", "dog-19", "guitar-3",
        "guitar-8",
        "guitar-10", "guitar-16", "person-1", "person-5", "person-10", "person-12", "pig-2", "pig-10", "pig-13",
        "pig-18",
        "rubicCube-1", "rubicCube-6", "rubicCube-14", "rubicCube-19", "swing-10", "swing-14", "swing-17", "swing-20",
        "drone-13", "drone-15", "drone-2", "drone-7", "pool-12", "pool-15", "pool-3", "pool-7", "rabbit-10",
        "rabbit-13",
        "rabbit-17", "rabbit-19", "racing-10", "racing-15", "racing-16", "racing-20", "robot-1", "robot-19", "robot-5",
        "robot-8", "sepia-13", "sepia-16", "sepia-6", "sepia-8", "sheep-3", "sheep-5", "sheep-7", "sheep-9",
        "skateboard-16", "skateboard-19", "skateboard-3", "skateboard-8", "tank-14", "tank-16", "tank-6", "tank-9",
        "tiger-12", "tiger-18", "tiger-4", "tiger-6", "train-1", "train-11", "train-20", "train-7", "truck-16",
        "truck-3",
        "truck-6", "truck-7", "turtle-16", "turtle-5", "turtle-8", "turtle-9", "umbrella-17", "umbrella-19",
        "umbrella-2",
        "umbrella-9", "yoyo-15", "yoyo-17", "yoyo-19", "yoyo-7", "zebra-10", "zebra-14", "zebra-16", "zebra-17",
        "elephant-1", "elephant-12", "elephant-16", "elephant-18", "goldfish-3", "goldfish-7", "goldfish-8",
        "goldfish-10",
        "hat-1", "hat-2", "hat-5", "hat-18", "kite-4", "kite-6", "kite-10", "kite-15", "motorcycle-1", "motorcycle-3",
        "motorcycle-9", "motorcycle-18", "mouse-1", "mouse-8", "mouse-9", "mouse-17", "flag-3", "flag-9", "flag-5",
        "flag-2", "frog-3", "frog-4", "frog-20", "frog-9", "gametarget-1", "gametarget-2", "gametarget-7",
        "gametarget-13",
        "hand-2", "hand-3", "hand-9", "hand-16", "helmet-5", "helmet-11", "helmet-19", "helmet-13", "licenseplate-6",
        "licenseplate-12", "licenseplate-13", "licenseplate-15", "electricfan-1", "electricfan-10", "electricfan-18",
        "electricfan-20", "chameleon-3", "chameleon-6", "chameleon-11", "chameleon-20", "crocodile-3", "crocodile-4",
        "crocodile-10", "crocodile-14", "gecko-1", "gecko-5", "gecko-16", "gecko-19", "fox-2", "fox-3", "fox-5",
        "fox-20",
        "giraffe-2", "giraffe-10", "giraffe-13", "giraffe-15", "gorilla-4", "gorilla-6", "gorilla-9", "gorilla-13",
        "hippo-1", "hippo-7", "hippo-9", "hippo-20", "horse-1", "horse-4", "horse-12", "horse-15", "kangaroo-2",
        "kangaroo-5", "kangaroo-11", "kangaroo-14", "leopard-1", "leopard-7", "leopard-16", "leopard-20", "lion-1",
        "lion-5", "lion-12", "lion-20", "lizard-1", "lizard-3", "lizard-6", "lizard-13", "microphone-2", "microphone-6",
        "microphone-14", "microphone-16", "monkey-3", "monkey-4", "monkey-9", "monkey-17", "shark-2", "shark-3",
        "shark-5",
        "shark-6", "squirrel-8", "squirrel-11", "squirrel-13", "squirrel-19", "volleyball-1", "volleyball-13",
        "volleyball-18", "volleyball-19"
    ]
    return lasot_test


if __name__ == '__main__':
    args = parse_args()
    cvt = build_converter(args.root_dir, args.downsample)
    nthread = args.nthread if args.nthread > 0 else None
    cvt.process(args.output_dir, instanc_size=args.size, num_thread=nthread)


