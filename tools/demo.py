# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths
import torch
import os
import cv2
import numpy as np
import argparse

from siam_tracker.utils import Config, load_checkpoint, img_np2tensor
from siam_tracker.utils import box as ubox
from siam_tracker.models import build_tracker


def parse_args():
    parser = argparse.ArgumentParser(description='Test a tracker.')
    parser.add_argument('--config',
                        default='',
                        type=str,
                        help='tracker configuration file path.')
    parser.add_argument('--video',
                        default='data/demo_video/BlurOwl.avi',
                        type=str,
                        help='demo video path')
    parser.add_argument('--box',
                        default='349,193,408,298',
                        type=str,
                        help='initialization bounding box. (in format of (x1, y1, x2, y2))')
    parser.add_argument('--checkpoint',
                        default='',
                        type=str,
                        help='checkpoint path. If it is empty, we will search the path defined in '
                             'tracker configuration file.')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='cpu mode')
    parser.add_argument('--output',
                        default='',
                        type=str,
                        help='output video path. If it is empty, the demo video will be generated '
                             'in the same folder of input video.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # load configuration file
    cfg = Config.fromfile(args.config)
    # build tracker
    tracker = build_tracker(cfg.model, test_cfg=cfg.test_cfg, is_training=False)
    if args.checkpoint == '':
        args.checkpoint = os.path.join(cfg.work_dir, 'latest.pth')
    load_checkpoint(tracker, args.checkpoint)
    if not args.cpu:
        tracker.cuda()
    # convert initialization box from (x1, y1, x2, y2) to (xc, yc, w, h)
    init_box = torch.FloatTensor(list(map(float, args.box.split(','))))
    init_box = ubox.xyxy_to_xcycwh(init_box)

    if args.output == '':
        args.output = os.path.join(os.path.dirname(args.video), 'demo_out.mp4')

    # load videos
    cap = cv2.VideoCapture(args.video)
    writer = None
    frame_count = 0
    while cap.isOpened():
        # load each frame from the video
        _, frame = cap.read()
        if frame is None:
            break
        # convert numpy array to torch tensor.
        img_tensor = img_np2tensor(frame)
        if not args.cpu:
            img_tensor = img_tensor.cuda()
        if frame_count == 0:
            # for the first frame, we use the given bounding box to initialize tracker.
            tracker.initialize(img_tensor, init_box)
            box = init_box
        else:
            # for the rest frames, we directly predict the coordinates of object box.
            box = tracker.predict(img_tensor)
        # draw output box on the original frame
        box_np = ubox.xcycwh_to_xyxy(box).numpy()
        box_np = np.round(box_np).astype(int)
        frame = cv2.rectangle(frame, (box_np[0], box_np[1]), (box_np[2], box_np[3]), color=(63, 220, 32), thickness=2)
        if writer is None:
            w, h = frame.shape[1], frame.shape[0]
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
        writer.write(frame)
        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    print('Done! The output video is saved to {}'.format(args.output))
