# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.


import torch


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None, use_torch=True):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.use_torch = use_torch
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        if self.use_torch:
            h_ratios = torch.sqrt(self.ratios)
            w_ratios = 1 / h_ratios
            if self.scale_major:
                ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
                hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
            else:
                raise NotImplementedError

            base_anchors = torch.stack(
                [
                    x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
                ],
                dim=-1).round()
        else:
            size = w * h
            size_ratios = size / self.ratios
            ws = torch.sqrt(size_ratios).round()
            hs = (ws * self.ratios).round()
            ws = ws * self.scales[None, :]
            hs = hs * self.scales[None, :]

            base_anchors = torch.stack(
                [
                    x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
                ],
                dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_x = shift_x - (feat_w - 1) / 2.0 * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_y = shift_y - (feat_h - 1) / 2.0 * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors


if __name__ == '__main__':
    s = AnchorGenerator(8, [8., ], [0.33, 0.5, 1.0, 2.0, 3.0], ctr=(0, 0))
    # s.grid_anchors([17, 17], stride=8, device='cpu')