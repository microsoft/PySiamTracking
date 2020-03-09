import os
import cv2
import numpy as np
import argparse
import zipfile
import _pickle as pickle


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Visualization crop results')
    parser.add_argument('--annotation', dest='annotation',
                        help='annotation file path',
                        default='',
                        type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='image dir',
                        default='',
                        type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    with open(args.annotation, 'rb') as f:
        annos = pickle.load(f)

    inds = np.random.permutation(len(annos['seqs']))

    is_zip = False
    if args.image_dir[-4:] == '.zip':
        is_zip = True
        zf = zipfile.ZipFile(args.image_dir, 'r')

    for i in inds:
        seq = annos['seqs'][i]
        num_frames = len(seq['frames'])
        for frame_id in range(num_frames):
            if not is_zip:
                frame_path = os.path.join(args.image_dir, seq['frames'][frame_id])
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            else:
                buf = zf.read(name=seq['frames'][frame_id])
                img = cv2.imdecode(
                    np.fromstring(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
            bboxes = seq['bboxes'][frame_id]

            for bbox_id, bbox in enumerate(bboxes):
                ec = (0, 255, 0) if bbox_id == 0 else (0, 255, 255)
                bbox = np.round(bbox).astype(np.int32)
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=ec)
            cv2.imshow('show', img)
            cv2.waitKey(40)
        cv2.waitKey(0)
