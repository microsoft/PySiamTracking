import _init_paths
import argparse
import numpy as np
import os
import _pickle as pickle
from PIL import Image
import time
from preprocessing.converter import PairAnnotationConverter


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert YoutubeBB into cropped image pairs')
    parser.add_argument('--nthread', dest='nthread',
                        help='number of thread',
                        default=-1,
                        type=int)
    parser.add_argument('--annotation', dest='annotation',
                        help='annotation file path',
                        default='',
                        type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='image dir',
                        default='',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output dir',
                        default='.',
                        type=str)
    parser.add_argument('--name', dest='name',
                        help='dataset name',
                        default='youtube_bb_train',
                        type=str)
    parser.add_argument('--size', dest='size',
                        help='crop image size',
                        default=448,
                        type=int)

    args = parser.parse_args()
    return args


def str2anno(line):
    sp = line.split(',')
    vid = sp[0]
    timestamp = int(sp[1])
    category_id = int(sp[2])
    # classname = sp[3]
    obj_id = int(sp[4])
    is_absent = sp[5] == 'absent'
    x1, x2, y1, y2 = list(map(float, sp[6:10]))
    return vid, timestamp, category_id, obj_id, is_absent, x1, x2, y1, y2


def sort_by_timestamp(anno):
    timestamp_list = list(anno['annos'].keys())
    # sort timestamp list
    timestamp_inds = np.argsort(np.array(timestamp_list, dtype=np.int64))
    bboxes_list = [np.array(anno['annos'][timestamp_list[i]]) for i in timestamp_inds]
    timestamp_list = [timestamp_list[i] for i in timestamp_inds]
    return bboxes_list, timestamp_list


def extract_info_from_annotation(root_img_dir, anno):
    """ Extract information from annotation object (from load_annotation) """
    img = None
    raw_bboxes_list, timestamp_list = sort_by_timestamp(anno)

    track_list = []
    frame_list = []
    bboxes_list = []
    track_id_dict = {}

    for i, (raw_bboxes, timestamp) in enumerate(zip(raw_bboxes_list, timestamp_list)):
        _cat_id, _obj_id = int(raw_bboxes[0, 4]), int(raw_bboxes[0, 5])
        img_path = os.path.join(root_img_dir, '{}'.format(_cat_id),
                                '{vid}_{time}_{cat_id}_{obj_id}.jpg'.format(vid=anno['vid'],
                                                                            time=timestamp,
                                                                            cat_id=_cat_id,
                                                                            obj_id=_obj_id))
        if not os.path.exists(img_path):
            # print("[warning] cannot find image file {}.".format(img_path))
            continue
        frame_list.append(img_path)
        if img is None:
            img = Image.open(img_path)
        width, height = img.size
        bboxes = np.zeros((raw_bboxes.shape[0], 6), dtype=np.float32)
        bboxes[:, 0:4:2] = raw_bboxes[:, 0:4:2] * width
        bboxes[:, 1:4:2] = raw_bboxes[:, 1:4:2] * height
        bboxes[:, 4] = raw_bboxes[:, 4] + 1
        for bbox_id, raw_bbox in enumerate(raw_bboxes):
            category_id, obj_id = int(raw_bbox[4]), int(raw_bbox[5])
            if (category_id, obj_id) not in track_id_dict:
                track_id_dict[(category_id, obj_id)] = len(track_list)
                track_list.append([])
            track_id = track_id_dict[(category_id, obj_id)]
            track_list[track_id].append((i, bbox_id))
        bboxes_list.append(bboxes)

    video_info = dict(
        name=anno['vid'],
        frames=frame_list,
        bboxes=bboxes_list,
        frame_ids=timestamp_list,
        tracks=track_list
    )
    return video_info


def load_annotation(annotation_path):
    """ Load annotations from raw file.
    The return object is a list in which the echo element is a dict. The structure is
    {
        'vid' (str): video name
        'annos' (dict)[timestamp]: [(x1, y1, x2, y2, category_id, obj_id)]
    }
    """
    with open(annotation_path, 'r') as f:
        lines = f.read().splitlines()
    num_lines = len(lines)
    results = []
    current_video = {'vid': ''}
    for line_id, line in enumerate(lines):
        if line.strip() == '':
            continue
        (vid, timestamp, category_id, obj_id, is_absent, x1, x2, y1, y2) = str2anno(line)
        if vid != current_video['vid']:
            if 'annos' in current_video and len(current_video['annos']) > 0:
                results.append(current_video)
            current_video = {'vid': vid, 'annos': {}}
        annos = current_video['annos']

        if is_absent:
            continue
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2:
            continue

        if timestamp not in annos:
            annos[timestamp] = [(x1, y1, x2, y2, category_id, obj_id)]
        else:
            annos[timestamp].append((x1, y1, x2, y2, category_id, obj_id))
        if (line_id + 1) % 10000 == 0:
            print("Loading annotations {}/{}".format(line_id+1, num_lines))
    return results


if __name__ == '__main__':

    category_info = ['person', 'bird', 'bicycle', 'boat', 'bus',
                     'bear', 'cow', 'cat', 'giraffe', 'potted plant',
                     'horse', 'motorcycle', 'knife', 'airplane', 'skateboard',
                     'train', 'truck', 'zebra', 'toilet', 'dog',
                     'elephant', 'umbrella', 'none', 'car']

    args = parse_args()

    annos = load_annotation(args.annotation)
    cached_path = '{}_cached.pkl'.format(args.name)

    if os.path.exists(cached_path):
        with open(cached_path, 'rb') as f:
            video_dataset = pickle.load(f)
    else:
        video_dataset = PairAnnotationConverter(name=args.name, categories=category_info)
        failure_count = 0
        tic = time.time()
        for anno_id, anno in enumerate(annos):
            try:
                video_info = extract_info_from_annotation(args.image_dir, anno)
                if len(video_info['frames']) == 0:
                    failure_count += 1
                else:
                    video_dataset.add_video(video_info)
            except Exception:
                failure_count += 1

            if anno_id % 100 == 0:
                duration = time.time() - tic
                eta = duration / (anno_id + 1) * (len(annos) - anno_id - 1)
                print("Extract video information... {}/{}, [ETA {:.2f} hour]".format(anno_id, len(annos), eta / 3600.0))

        print("All {success} videos are available. ({success}/{all})".format(success=len(annos) - failure_count,
                                                                             all=len(annos)))
        # cache into file
        with open(cached_path, 'wb') as f:
            pickle.dump(video_dataset, f)

    nthread = args.nthread if args.nthread > 0 else None
    video_dataset.process(args.output_dir, instanc_size=args.size, num_thread=nthread)

    # if os.path.exists(cached_path):
    #     os.remove(cached_path)
