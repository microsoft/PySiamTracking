def get_data_config(z_size, x_size, data_root, storage_backend):
    train_data = dict(
        sample_per_epoch=320000,
        gray_prob=0.01,
        datasets=[
            dict(
                name='coco',
                data_root=data_root,
                storage=storage_backend,
                max_frame_dist=200,
                sample_type_prob=[(0.60, 0.20, 0.20)],
                max_category_num=100000,
                sample_weight=0.15
            ),
            dict(
                name='got10k',
                data_root=data_root,
                storage=storage_backend,
                max_frame_dist=200,
                sample_type_prob=[(0.9, 0.05, 0.05)],
                max_category_num=100000,
                sample_weight=0.30
            ),
            dict(
                name='lasot',
                data_root=data_root,
                storage=storage_backend,
                max_frame_dist=200,
                sample_type_prob=[(0.9, 0.05, 0.05)],
                max_category_num=100000,
                sample_weight=0.30
            ),
            dict(
                name='trackingnet',
                data_root=data_root,
                storage=storage_backend,
                max_frame_dist=200,
                sample_type_prob=[(0.9, 0.05, 0.05)],
                max_category_num=100000,
                sample_weight=0.25
            ),
        ],
        transforms=dict(
            image_z=[
                dict(type='ColorTransform',
                     brightness_prob=0.15, brightness_delta=15,
                     contrast_prob=0.15, contrast_delta=0.2,
                     hue_prob=0.1, hue_delta=12,
                     saturation_prob=0.1, saturation_delta=0.2),
                dict(type='ToTensor'),
                dict(type='RandomCropAndResize',
                     max_scale_ratio=0.02,
                     max_shift_pixel=4,
                     out_width=z_size,
                     out_height=z_size,
                     keep_ar=True)
            ],
            image_x=[
                dict(type='OcclusionTransform', occ_prob=0.1),
                dict(type='GeometryTransform', flip_prob=0.25, rotation_prob=0.0),
                dict(type='ColorTransform',
                     brightness_prob=0.15, brightness_delta=30,
                     contrast_prob=0.15, contrast_delta=0.3,
                     hue_prob=0.1, hue_delta=6,
                     saturation_prob=0.1, saturation_delta=0.15),
                dict(type='BlurTransform',
                     blur_prob=0.25,
                     gaussian_ksize=[3, 5, 7, 9, 11, 13, 15, 17, 19, ],
                     gaussian_ksize_prob=[0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
                     average_ksize=[3, 5, 7, 9],
                     average_ksize_prob=[0.4, 0.3, 0.2, 0.1],
                     average_prob=0.3,
                     downsample_ratio=[0.2, 0.25, 0.33, 0.5],
                     downsample_ratio_prob=[0.1, 0.2, 0.3, 0.4],
                     motion_ksize=[5, 7, 9, 11, 13, 15, 17, 19],
                     motion_ksize_prob=[0.12, 0.15, 0.20, 0.14, 0.13, 0.10, 0.09, 0.07]),
                dict(type='ToTensor'),
                dict(type='RandomCropAndResize',
                     max_scale_ratio=0.18,
                     max_shift_pixel=64,
                     out_width=x_size,
                     out_height=x_size,
                     keep_ar=False)
            ],
            video_z=[
                dict(type='ColorTransform',
                     brightness_prob=0.1, brightness_delta=15,
                     contrast_prob=0.1, contrast_delta=0.2,
                     hue_prob=0.0, hue_delta=12,
                     saturation_prob=0.0, saturation_delta=0.2),
                dict(type='ToTensor'),
                dict(type='RandomCropAndResize',
                     max_scale_ratio=0.02,
                     max_shift_pixel=4,
                     out_width=z_size,
                     out_height=z_size,
                     keep_ar=True)
            ],
            video_x=[
                dict(type='OcclusionTransform', occ_prob=0.1),
                dict(type='GeometryTransform', flip_prob=0.1, rotation_prob=0.0),
                dict(type='ColorTransform',
                     brightness_prob=0.15, brightness_delta=30,
                     contrast_prob=0.15, contrast_delta=0.3,
                     hue_prob=0.1, hue_delta=6,
                     saturation_prob=0.1, saturation_delta=0.15),
                dict(type='ToTensor'),
                dict(type='RandomCropAndResize',
                     max_scale_ratio=0.18,
                     max_shift_pixel=64,
                     out_width=x_size,
                     out_height=x_size,
                     keep_ar=False)
            ]
        )
    )

    return train_data
