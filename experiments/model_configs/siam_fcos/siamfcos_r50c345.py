from experiments.data_configs.got10k_lasot_trackingnet_coco import get_data_config

data_root = 'data/benchmark'
work_dir = './output/siamfcos/siamfcos_r50c345/'
log_level = 'INFO'
resume_from = None
cudnn_benchmark = True
model = dict(
    type='SiamFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=5,
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 1, 1),
        stem_padding=0,
        norm_eval=False,
        pretrained='data/pretrained_models/resnet50.pth',
        init_type='xavier_uniform',
    ),
    neck=None,
    fusion=dict(
        type='MultiLevelFusion',
        fusions=[
            dict(
                type='CrossCorrelation',
                feat_name='conv3',
                in_channels=512,
                corr_channels=128,
                out_channels=128,
                depth_wise=True,
                pre_kernel_size=3,
                init_type='xavier_uniform'
            ),
            dict(
                type='CrossCorrelation',
                feat_name='conv4',
                in_channels=1024,
                corr_channels=256,
                out_channels=256,
                depth_wise=True,
                pre_kernel_size=3,
                init_type='xavier_uniform'
            ),
            dict(
                type='CrossCorrelation',
                feat_name='conv5',
                in_channels=2048,
                corr_channels=512,
                out_channels=512,
                depth_wise=True,
                pre_kernel_size=3,
                init_type='xavier_uniform'
            )
        ]
    ),
    head=dict(
        type='FCOSHead',
        stride=[8, 8, 8],
        target_means=(32.0, 32.0, 32.0, 32.0),
        target_stds=(8.0, 8.0, 8.0, 8.0),
        pre_convs=[
            dict(num_layers=2, in_channels=128, out_channels=256, kernel_size=3, padding=0, nonlinear_last=True),
            dict(num_layers=2, in_channels=256, out_channels=256, kernel_size=3, padding=0, nonlinear_last=True),
            dict(num_layers=2, in_channels=512, out_channels=256, kernel_size=3, padding=0, nonlinear_last=True),
        ],
        head_convs=[
            [
                dict(num_layers=1, in_channels=256, out_channels=1, kernel_size=1, nonlinear_last=False),
                dict(num_layers=1, in_channels=256, out_channels=4, kernel_size=1, nonlinear_last=False),
            ],
            [
                dict(num_layers=1, in_channels=256, out_channels=1, kernel_size=1, nonlinear_last=False),
                dict(num_layers=1, in_channels=256, out_channels=4, kernel_size=1, nonlinear_last=False),
            ],
            [
                dict(num_layers=1, in_channels=256, out_channels=1, kernel_size=1, nonlinear_last=False),
                dict(num_layers=1, in_channels=256, out_channels=4, kernel_size=1, nonlinear_last=False),
            ]
        ],
        multi_level_learnable_weights=True,
        cls_loss=dict(type='BinaryCrossEntropyLoss', loss_weight=1.0),
        reg_loss=dict(type='L1Loss', loss_weight=1.0),
        init_type='xavier_uniform'
    ),
)
test_cfg = dict(
    z_size=127,
    x_size=255,
    z_feat_size=6,
    x_feat_size=26,
    search_region=dict(num_scales=1, scale_step=1.0, context_amount=0.5),
    min_box_size=8,
    penalty_k=0.2,
    min_score_threshold=0.1,
    linear_inter_rate=0.5,
    window=dict(weight=0.40),
)
train_cfg = dict(
    type='PairwiseWrapper',
    samples_per_gpu=8,
    workers_per_gpu=3,
    z_size=255,
    x_size=255,
    z_feat_size=6,
    x_feat_size=26,
    num_freeze_blocks=0,
    fcos=dict(
        assigner=dict(type='FoveaPointAssigner', sigma1=0.60, sigma2=0.90),
        sampler=dict(type='RandomSampler', num=64, pos_fraction=0.25, neg_pos_ub=-1),
        pos_weight=-1,
    ),
    # optimizer
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    optimizer_config=dict(
        grad_clip=dict(max_norm=20, norm_type=2),
        optimizer_cfg=dict(
            type='SGD',
            params=[dict(name='head/weight'), dict(name='head/bias', weight_decay=0.0)],
            lr=0.01,
            weight_decay=0.0001
        ),
        optimizer_schedule=[
            dict(
                start_epoch=1,
                type='SGD',
                params=[dict(name='all/weight'), dict(name='all/bias', weight_decay=0.0)],
                lr=0.01,
                weight_decay=0.0001
            )
        ]
    ),
    # learning policy
    lr_config=dict(
        policy='step',
        gamma=0.1,
        step=[25, 45],
        warmup='linear',
        warmup_iters=100,
        warmup_ratio=0.1 / 3,
    ),
    checkpoint_config=dict(interval=1),
    log_config=dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook'),
        ]
    ),
    total_epochs=50,
    workflow=[('train', 1)],
)

storage_backend = dict(type='ZipWrapper', cache_into_memory=True)
train_cfg['train_data'] = get_data_config(train_cfg['z_size'], train_cfg['x_size'], data_root, storage_backend)
