import os
import numpy as np
import _pickle as pickle
import cv2
import time
import datetime
import multiprocessing


class PairAnnotationConverter(object):

    def __init__(self, name, categories):
        """ According to SiamFC[1] or SiamRPN[2], the input videos will be processed so that they can
        be efficiently used as image pairs. This class is used to generate the annotations which is
        suitable for training. Echo frame will be center cropped w.r.t the ground-truth bounding boxes.

        [1]: Fully-Convolutional Siamese Networks for Object Tracking
        [2]: High Performance Visual Tracking with Siamese Region Proposal Network

        Args:
            name (str): the dataset name
            categories (list[str]): the name list of categories.

        """
        self.name = name
        if isinstance(categories[0], str):
            self.categories = [dict(id=i+1, name=name) for i, name in enumerate(categories)]
        else:
            self.categories = categories
        self.videos = []

    def add_video(self, video):
        """
        Add video annotation into waiting list.

        video annotation is a dict, whose keys are shown as following:
        {
            'name': str,
            'frames': [(list of str)],
            'bboxes': [(list of numpy array)], [x1, y1, x2, y2, class_id, ignore_flag]
            'frame_ids': [list of int],
            'tracks': [[(list of tuple)]], echo tuple has two values, the first one is frame index and the next one
                      is box index.
        }
        """
        self.videos.append(video)

    def process(self, output_dir, instanc_size, num_thread=None):
        """ Process echo video: generate annotations and crop images.

        Args:
            output_dir (str): the output directory.
            instanc_size (int): the cropped size, a.k.a, search region size. (the template size will always be 127)
            num_thread (int): the number of thread to process the dataset.

        Returns:
            None

        """
        print("All {} videos".format(len(self.videos)))
        num_trks = sum([len(v['tracks']) for v in self.videos])
        noti_interval = max(num_trks // 18, 100)
        print("All {} tracks".format(num_trks))
        seq_annos = []
        seq_count = 0
        tasks = []
        for vid, video in enumerate(self.videos):
            for tid, track in enumerate(video['tracks']):
                # extract category id from the first frame in this tracklet.
                category_id = int(video['bboxes'][track[0][0]][track[0][1], 4])
                track_bboxes = []
                track_frame_ids = []
                track_frame_paths = []
                for i in range(len(track)):
                    fid, bid = track[i]  # frame index & bounding box index
                    bboxes = video['bboxes'][fid].copy()
                    # the core box is always in the first place.
                    if bid != 0:
                        inds = list(range(0, len(bboxes)))
                        inds[bid] = 0
                        inds[0] = bid
                        bboxes = bboxes[inds]
                    assert category_id == int(bboxes[0, 4]), \
                        "Mismatch category id in {} {}".format(video['name'], tid)

                    # generate search region
                    crop_bbox = generate_search_region(bboxes[0, 0:4], instanc_size=instanc_size)
                    # project bboxes' coordinates into search region.
                    bboxes = bbox_in_search_region(bboxes, crop_bbox, instanc_size)

                    frame_id = int(video['frame_ids'][fid])
                    # video_name, track_id, frame_id
                    frame_path = '{}_{}_{}.jpg'.format(video['name'], tid, frame_id)
                    track_bboxes.append(bboxes)
                    track_frame_ids.append(frame_id)
                    track_frame_paths.append(frame_path)

                    # add into task list (source, destination, x1, y1, x2, y2)
                    tasks.append((
                        video['frames'][fid], frame_path, crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]
                    ))

                seq_annos.append(dict(
                    frame_id=track_frame_ids,
                    bboxes=track_bboxes,
                    frames=track_frame_paths,
                    category=category_id
                ))
                seq_count += 1
                if seq_count % noti_interval == 0:
                    print("Generate annotations {}/{}".format(seq_count, num_trks))

        img_output_dir = os.path.join(output_dir, 'images')
        if not os.path.isdir(img_output_dir):
            os.makedirs(img_output_dir)

        annos = dict(categories=self.categories, seqs=seq_annos)
        # save annotation into file.
        with open(os.path.join(output_dir, '{}.pkl'.format(self.name)), 'wb') as f:
            pickle.dump(annos, f)

        print("Start generating cropped images. (All {} images)".format(len(tasks)))
        if num_thread is None:
            num_thread = multiprocessing.cpu_count()
        print("Number of thread: {}".format(num_thread))
        if num_thread == 1:
            # single thread
            generate_crop_image(tasks, img_output_dir, instanc_size)
        else:
            job_list = []
            bound = [int(len(tasks) * i / num_thread) for i in range(num_thread+1)]
            for i in range(num_thread):
                job_list.append(multiprocessing.Process(
                    target=generate_crop_image,
                    args=(tasks[bound[i]:bound[i+1]], img_output_dir, instanc_size, i)
                ))
            for i in range(num_thread):
                job_list[i].start()
            for i in range(num_thread):
                job_list[i].join()
            for i in range(num_thread):
                job_list[i].terminate()

        valid_seq_annos = []
        for seq_anno in seq_annos:
            valid_flag = True
            for frame_path in seq_anno['frames']:
                if not os.path.exists(os.path.join(img_output_dir, frame_path)):
                    valid_flag = False
                    break
            if valid_flag:
                valid_seq_annos.append(seq_anno)

        if len(valid_seq_annos) != len(seq_annos):
            print("Filter invalid video {}-->{}".format(len(seq_annos), len(valid_seq_annos)))
        annos = dict(categories=self.categories, seqs=valid_seq_annos)
        # save annotation into file.
        with open(os.path.join(output_dir, '{}_clean.pkl'.format(self.name)), 'wb') as f:
            pickle.dump(annos, f)


def bbox_in_search_region(bboxes, crop_bbox, instanc_size):
    crop_size = [crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]]
    # assert crop_size[0] == crop_size[1]
    scale = instanc_size / crop_size[0]
    bboxes[:, 0:4:2] -= crop_bbox[0]
    bboxes[:, 1:4:2] -= crop_bbox[1]
    bboxes[:, 0:4] *= scale
    return bboxes


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def generate_search_region(bbox, context_amount=0.5, exemplar_size=127, instanc_size=255):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    crop_bbox = pos_s_2_bbox(target_pos, s_x)
    return crop_bbox


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def generate_crop_image(tasks, output_dir, out_sz, thread_id=0):
    num_tasks = len(tasks)
    tic = time.time()
    noti_interval = 5000
    for i, task in enumerate(tasks):
        src, dst, x1, y1, x2, y2 = task
        dst = os.path.join(output_dir, dst)
        if os.path.exists(dst):
            continue
        src_img = cv2.imread(src, cv2.IMREAD_COLOR)
        if src_img is None:
            continue
        if len(src_img.shape) < 3:
            continue
        avg_chans = np.mean(src_img, axis=(0, 1))
        dst_img = crop_hwc(src_img, (x1, y1, x2, y2), out_sz, padding=avg_chans)
        cv2.imwrite(dst, dst_img)
        if (i+1) % noti_interval == 0:
            toc = time.time()
            eta_sec = int((toc - tic) / noti_interval * (num_tasks - i - 1))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            print("[Thread {}] Cropping image {}/{}. ETA {}".format(thread_id, i, num_tasks, eta_str))
            tic = time.time()
