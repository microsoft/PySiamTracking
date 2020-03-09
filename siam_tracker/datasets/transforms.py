# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Data augmentation for training """
import torch
import numpy as np
import numpy.random as npr
import cv2

from ..utils.registry import Registry
from ..utils.crop import roi_crop
from ..utils.box import xcycwh_to_xyxy, bbox_overlaps

TRANSFORMS = Registry(name='transform')


def build_transforms(transform_cfgs):
    assert isinstance(transform_cfgs, list)
    transforms = []
    for transform_cfg in transform_cfgs:
        _cfg = transform_cfg.copy()
        transform_type = _cfg.pop('type')
        transforms.append(TRANSFORMS.get_module(transform_type)(**_cfg))
    transform_compose = Compose(transforms)
    return transform_compose


class Compose(object):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transforms):
        if isinstance(transforms, list):
            for t in transforms:
                self.transforms.append(t)
        else:
            self.transforms.append(transforms)

    def __call__(self, img, bbox):
        """ process images by different transformations.
        Args:
            img, numpy.ndarray or list of numpy.ndarray
            bbox, numpy.ndarray or list of numpy.ndarray
        """
        for t in self.transforms:
            img, bbox = t(img, bbox)
        return img, bbox


@TRANSFORMS.register_module
class OcclusionTransform(object):
    """ Manually add occlusion to the target object.
    The occlusion patch could be random colors or image patch from other positions.
    """
    def __init__(self,
                 occ_prob=0.2,
                 shift_range=64,
                 occ_range=(36, 96),
                 alpha_range=(0.6, 1.0),
                 inter_range=(0.05, 0.5),
                 outside_range=(-1.0, 0.00)):
        """
        Args:
            occ_prob (float): the probability that occlusion happens.
            shift_range (int): the max distance from occlusion center to target cetner.
            occ_range (tuple): the size range of occlusion patches
            alpha_range (tuple): the transparency rate range
            inter_range (tuple): if IoUs between occlusion patch and target patch address in this range
                                 the occlusion patch will be considered as 'intersection'
            outside_range (tuple): if IoUs between occlusion patch and target patch address in this range
                                 the occlusion patch will be considered as 'outside'
        """
        self.occ_prob = occ_prob
        self.shift_range = shift_range
        self.occ_range = occ_range
        self.inter_range = inter_range
        self.outside_range = outside_range
        self.alpha_range = alpha_range

    def collect_available_region(self,
                                 boxes,
                                 img_shape,
                                 intersection=False,
                                 num_candidates=500):
        """ Collect available occlusion patch coordinates. We random enumerate many boxes and select
        the valid one.

        Args:
            boxes (np.ndarray): in shape of [N, 4], the target bounding boxes
            img_shape (tuple): image shape [H, W]
            intersection (bool): if the occlusion region should be intersected with the target box.
            num_candidates (int): how many boxes we enumerate.
        """
        shifts = npr.uniform(-self.shift_range, self.shift_range, size=(num_candidates, 2))
        ctrs = (boxes[0:1, 0:2] + boxes[0:1, 2:4]) / 2.0 + shifts
        wh = npr.uniform(self.occ_range[0], self.occ_range[1], size=(num_candidates, 2))
        candidates = xcycwh_to_xyxy(np.hstack((ctrs, wh)))
        # clip to img shape
        candidates = np.round(candidates)
        candidates[:, 0:4:2] = np.clip(candidates[:, 0:4:2], a_min=0, a_max=img_shape[1])
        candidates[:, 1:4:2] = np.clip(candidates[:, 1:4:2], a_min=0, a_max=img_shape[0])
        # filter some invalid boxes
        wh = candidates[:, 2:4] - candidates[:, 0:2]
        valid_ind = np.where(np.all(wh > 4, axis=1))[0]
        if len(valid_ind) == 0:
            return None
        if len(valid_ind) < len(candidates):
            candidates = candidates[valid_ind]
        # calculate the IoU
        ious = bbox_overlaps(np.ascontiguousarray(boxes[0:1, 0:4]), candidates, mode='iof')
        max_ious = ious.max(axis=0)
        if intersection:
            min_th, max_th = self.inter_range
        else:
            min_th, max_th = self.outside_range
        valid_ind = np.where((max_ious >= min_th) & (max_ious <= max_th))[0]
        if len(valid_ind) == 0:
            return None
        idx = npr.choice(valid_ind)
        return candidates[idx, 0:4].astype(int)

    def __call__(self, img, boxes):
        # if there is no any boxes, just return the raw image & boxes.
        if boxes is None or len(boxes) == 0:
            return img, boxes
        if self.occ_prob > 0.0 and npr.rand() < self.occ_prob:
            has_inter = float(npr.rand()) < 0.5
            occ_box = self.collect_available_region(boxes, img.shape, has_inter)
            # if no valid occlusion box exists, return
            if occ_box is None:
                return img, boxes

            fill_type = npr.choice(2)
            alpha = npr.uniform(low=self.alpha_range[0], high=self.alpha_range[1])
            if fill_type == 0:
                # fill with random color
                color_mean = np.mean(img, axis=(0, 1))
                random_color = color_mean + npr.uniform(-32, 32, size=(3, ))
                occ_patch = np.clip(random_color, a_min=0, a_max=255.0).reshape(1, 1, 3)
            else:
                # fill with other patch
                crop_box = self.collect_available_region(boxes, img.shape, False)
                if crop_box is None:
                    return img, boxes
                x1, y1, x2, y2 = crop_box
                w, h = occ_box[2] - occ_box[0], occ_box[3] - occ_box[1]
                occ_patch = cv2.resize(img[y1:y2, x1:x2, :], dsize=(w, h))
            x1, y1, x2, y2 = occ_box
            img[y1:y2, x1:x2, :] = (1 - alpha) * img[y1:y2, x1:x2, :] + alpha * occ_patch
        return img, boxes


@TRANSFORMS.register_module
class GeometryTransform(object):

    def __init__(self,
                 flip_prob=0.0,
                 rotation_prob=0.0,
                 rotaion_angle=15):
        """ Geometry transformation, including flip & rotation. """

        # flip transform
        self.flip_prob = flip_prob

        # rotation transform
        self.rotation_prob = rotation_prob
        self.rotation_angle = rotaion_angle

    def __call__(self, img, boxes):
        """ Apply geometry transformation.
        Args:
            img (np.ndarray), [H, W, 3] the input image
            boxes: (np.ndarray), [N, 4] the bounding boxes
        Returns:
            img: (np.ndarray), [H, W, 3] the output image
            boxes: (np.ndarray), [N, 4] the bounding boxes after transformation.
        """
        boxes_cp = boxes.copy()
        # random flip
        if self.flip_prob > 0. and npr.rand() < self.flip_prob:
            img_width = img.shape[1]
            img = np.ascontiguousarray(img[:, ::-1, :])
            boxes_cp[:, 0] = img_width - boxes[:, 2] - 1
            boxes_cp[:, 2] = img_width - boxes[:, 0] - 1
        # random rotation
        if self.rotation_prob > 0. and npr.rand() < self.rotation_prob:
            ctr_x, ctr_y = img.shape[1] / 2.0, img.shape[0] / 2.0
            angle = npr.uniform(-self.rotation_angle, self.rotation_angle)
            rot_mat = cv2.getRotationMatrix2D((ctr_x, ctr_y), angle, scale=1.0)
            img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
            # bounding boxes coordinates
            num = boxes.shape[0]
            boxes_pts = self.get_box_points(boxes).reshape(-1, 2)
            boxes_pts = self.warp_points(boxes_pts, rot_mat).reshape(num, -1)
            boxes_cp[:, 0] = np.min(boxes_pts[:, 0::2], axis=1)
            boxes_cp[:, 1] = np.min(boxes_pts[:, 1::2], axis=1)
            boxes_cp[:, 2] = np.max(boxes_pts[:, 0::2], axis=1)
            boxes_cp[:, 3] = np.max(boxes_pts[:, 1::2], axis=1)
        return img, boxes_cp

    @staticmethod
    def get_box_points(boxes):
        """ Get box representative points.
        Args:
            boxes (np.ndarray): in shape of [M, 4]
        Returns:
            pts (np.ndarray): in shape of [M, K * 2]
        """
        ctrs = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0
        wh = (boxes[:, 2:4] - boxes[:, 0:2]) / 2.0
        wh = np.clip(wh, a_min=1.0, a_max=256)
        x = np.arange(-1, 1.01, 0.1, dtype=np.float32).reshape(1, -1) * wh[:, 0].reshape(-1, 1)  # [M, K]
        y = np.sqrt(np.clip(1 - (x / wh[:, 0]) ** 2, 0.0, 1.0)) * wh[:, 1].reshape(-1, 1)  # [M, K]
        pos_pts = np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis]), axis=2)
        neg_pts = np.concatenate((x[:, :, np.newaxis], -y[:, :, np.newaxis]), axis=2)
        pts = np.concatenate((pos_pts, neg_pts), axis=1)   # [M, K, 2]
        pts = pts + ctrs[:, np.newaxis, :]
        return pts

    @staticmethod
    def warp_points(pts, mat):
        """ Warp point by proj matrix.
        Args:
            pts (np.ndarray): in shape of [N, 2]
            mat (np.ndarray): in shape of [2, 3], projection matrix
        """
        pts_ext = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
        pts_ext = mat.dot(pts_ext.T).T
        return pts_ext


@TRANSFORMS.register_module
class ColorTransform(object):
    def __init__(self,
                 brightness_prob,
                 brightness_delta,
                 contrast_prob,
                 contrast_delta,
                 hue_prob,
                 hue_delta,
                 saturation_prob,
                 saturation_delta):

        # brightness transform
        self.brightness_prob = brightness_prob
        self.brightness_delta = brightness_delta

        # contrast transform
        self.contrast_prob = contrast_prob
        self.contrast_delta = contrast_delta

        # hue transform
        self.hue_prob = hue_prob
        self.hue_delta = hue_delta

        # saturation transform
        self.saturation_prob = saturation_prob
        self.saturation_delta = saturation_delta

    def __call__(self, img, boxes):
        img_cp = img.astype(np.float32, copy=True)
        # Random brightness
        if self.brightness_prob > 0. and npr.rand() < self.brightness_prob:
            img_cp += npr.uniform(-self.brightness_delta, +self.brightness_delta)

        # Random contrast
        if self.contrast_prob > 0. and npr.rand() < self.contrast_prob:
            img_cp *= np.exp(npr.uniform(-self.contrast_delta, self.contrast_delta))

        # Random hue & saturation
        has_hue = self.hue_prob > 0. and npr.rand() < self.hue_prob
        has_saturation = self.saturation_prob > 0. and npr.rand() < self.saturation_prob

        if has_hue or has_saturation:
            # convert to HSV color space

            # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range
            # is [0,255]. Different software use different scales. So if you are comparing
            #  OpenCV values with them, you need to normalize these ranges.
            uint_img = self.cvt_float_im_to_uint8(img_cp)
            hsv_img = cv2.cvtColor(uint_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            if has_hue:
                hsv_img[:, :, 0] += npr.uniform(-self.hue_delta, self.hue_delta)
            if has_saturation:
                hsv_img[:, :, 1] *= np.exp(npr.uniform(-self.saturation_delta, self.saturation_delta))

            img_cp = cv2.cvtColor(self.cvt_float_im_to_uint8(hsv_img, is_bgr=False), cv2.COLOR_HSV2BGR)
        else:
            img_cp = self.cvt_float_im_to_uint8(img_cp)

        return img_cp, boxes

    @staticmethod
    def cvt_float_im_to_uint8(img, is_bgr=True):
        """ convert data type from numpy.float32 to numpy.uint8 """
        nimg = np.round(np.clip(img, 0, 255)).astype(np.uint8)
        if not is_bgr:
            nimg[:, :, 0] = np.clip(nimg[:, :, 0], 0, 179)
        return nimg


@TRANSFORMS.register_module
class BlurTransform(object):
    def __init__(self,
                 blur_prob,
                 gaussian_prob=None,
                 gaussian_ksize=None,
                 gaussian_ksize_prob=None,
                 average_prob=None,
                 average_ksize=None,
                 average_ksize_prob=None,
                 downsample_prob=None,
                 downsample_ratio=None,
                 downsample_ratio_prob=None,
                 motion_prob=None,
                 motion_ksize=None,
                 motion_ksize_prob=None):

        self.blur_prob = blur_prob

        # Gaussian blur
        self.gaussian_prob = gaussian_prob if gaussian_prob is not None else 1
        self.gaussian_ksize = gaussian_ksize if gaussian_ksize is not None else np.arange(3, 19, 2, dtype=int)
        self.gaussian_ksize_prob = gaussian_ksize_prob \
            if gaussian_ksize_prob is not None else np.ones(len(self.gaussian_ksize)) / len(self.gaussian_ksize)

        # Average blur
        self.average_prob = average_prob if average_prob is not None else 1
        self.average_ksize = average_ksize if average_ksize is not None else np.arange(3, 11, 2, dtype=int)
        self.average_ksize_prob = average_ksize_prob \
            if average_ksize_prob is not None else np.ones(len(self.average_ksize)) / len(self.average_ksize)

        # Downsample-upsample blur (mimic low-resolution case)
        self.downsample_prob = downsample_prob if downsample_prob is not None else 1
        self.downsample_ratio = downsample_ratio \
            if downsample_ratio is not None else np.array([0.2, 0.25, 0.33, 0.5], dtype=np.float32)
        self.downsample_ratio_prob = downsample_ratio_prob \
            if downsample_ratio_prob is not None else np.ones(len(self.downsample_ratio)) / len(self.downsample_ratio)

        # Motion blur
        self.motion_prob = motion_prob if motion_prob is not None else 1
        self.motion_ksize = motion_ksize if motion_ksize is not None else np.arange(5, 19, 2, dtype=int)
        self.motion_ksize_prob = motion_ksize_prob \
            if motion_ksize_prob is not None else np.ones(len(self.motion_ksize)) / len(self.motion_ksize)

        self.type_prob = np.array([self.gaussian_prob,
                                   self.average_prob,
                                   self.downsample_prob,
                                   self.motion_prob], dtype=np.float32)
        assert self.type_prob.sum() > 0
        self.type_prob = self.type_prob / self.type_prob.sum()

    def __call__(self, img, boxes):
        if npr.rand() >= self.blur_prob:
            return img, boxes
        type_id = npr.choice(4, p=self.type_prob)
        # 0: gaussian blur
        # 1: average blur
        # 2: downsample-upsample blur
        # 3: motion blur
        if type_id == 0:
            k = npr.choice(self.gaussian_ksize, p=self.gaussian_ksize_prob)
            img = cv2.GaussianBlur(img, (k, k), 0)
        elif type_id == 1:
            k = npr.choice(self.average_ksize, p=self.average_ksize_prob)
            img = cv2.blur(img, (k, k))
        elif type_id == 2:
            img_height, img_width, _ = img.shape
            fx = npr.choice(self.downsample_ratio, p=self.downsample_ratio_prob)
            img = cv2.resize(cv2.resize(img, dsize=None, fx=fx, fy=fx), (img_width, img_height))
        else:
            k = npr.choice(self.motion_ksize, p=self.motion_ksize_prob)
            kmat = self.get_blur_kernel(k)
            img = cv2.filter2D(img, -1, kmat, borderType=cv2.BORDER_REPLICATE)
        return img, boxes

    @staticmethod
    def get_blur_kernel(kernel_size):
        s = int(kernel_size)
        s2 = int(s // 2)
        d = npr.rand() * 2 - 1.
        # d = 0
        mat = np.zeros((s, s), np.float32)

        x_inds = np.arange(-s2, s2 + 1, 1, dtype=np.int32)
        r = np.tan(d * np.pi / 4)
        y_inds = np.round(x_inds.astype(np.float32, copy=False) * r).astype(np.int32)
        y_inds = np.clip(y_inds, -s2, s2)
        x_inds += s2
        y_inds += s2
        if npr.rand() < 0.5:
            mat[y_inds, x_inds] = 1.0 / len(x_inds)
        else:
            mat[x_inds, y_inds] = 1.0 / len(x_inds)
        return mat


@TRANSFORMS.register_module
class RandomCropAndResize(object):
    def __init__(self, max_scale_ratio, max_shift_pixel, out_width, out_height, keep_ar=False):
        self.max_scale_ratio = max_scale_ratio
        self.max_shift_pixel = max_shift_pixel
        self.out_width = out_width
        self.out_height = out_height
        self.keep_ar = keep_ar

    def __call__(self, img, bbox):
        assert isinstance(img, torch.Tensor) and isinstance(bbox, (list, torch.Tensor))
        if img.dim() == 3:
            img = img.unsqueeze(0)
        num_img = img.size(0)

        img_height, img_width = img.size(2), img.size(3)
        ctr_x, ctr_y = (img_width - 1) / 2.0, (img_height - 1) / 2.0
        crop_box = torch.tensor([[ctr_x, ctr_y, self.out_width, self.out_height]], dtype=torch.float32).repeat(num_img, 1)
        if self.max_shift_pixel > 0:
            xy_shift = torch.FloatTensor(num_img, 2).uniform_(-self.max_shift_pixel, self.max_shift_pixel)
            crop_box[:, 0:2] += xy_shift
        if self.max_scale_ratio > 0:
            wh_scale = torch.FloatTensor(num_img, 2).uniform_(-self.max_scale_ratio, self.max_scale_ratio)
            if self.keep_ar:
                wh_scale[:, 0] = wh_scale[:, 1]
            crop_box[:, 2:4] *= torch.exp(wh_scale)

        xyxy = xcycwh_to_xyxy(crop_box)
        crop_img_tensor = roi_crop(img, xyxy, out_height=self.out_height, out_width=self.out_width)
        if bbox is not None:
            bbox[:, 0:4:2] -= xyxy[0, 0]
            bbox[:, 1:4:2] -= xyxy[0, 1]
            bbox[:, 0:4:2] *= (float(self.out_width) / crop_box[0, 2])
            bbox[:, 1:4:2] *= (float(self.out_height) / crop_box[0, 3])
        return crop_img_tensor, bbox


@TRANSFORMS.register_module
class ToTensor(object):

    def __init__(self):
        pass

    def __call__(self, img, bbox):
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        bbox_tensor = torch.from_numpy(bbox).float()
        img_tensor = img_tensor.permute(0, 3, 1, 2).float().contiguous()
        return img_tensor, bbox_tensor
