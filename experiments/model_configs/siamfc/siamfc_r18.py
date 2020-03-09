from experiments.data_configs.siamfc_got10k_lasot_trackingnet_coco import get_data_config

data_root = 'data/benchmark'
work_dir = './output/siamfc/siamfc_r18_bn/'
log_level = 'INFO'
cudnn_benchmark = True
resume_from = None
model = dict(
    type='SiamFC',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        strides=(1, 2, 1),
        dilations=(1, 1, 1),
        stem_padding=0,
        norm_eval=False,
        pretrained='data/pretrained_models/resnet18.pth',
        init_type='xavier_uniform',
    ),
    fusion=dict(
        type='CrossCorrelation',
        feat_name='conv4',
        in_channels=256,
        corr_channels=256,
        out_channels=1,
        depth_wise=False,
        pre_kernel_size=1,
        share_pre_conv=True,
        init_type='xavier_uniform'
    ),
    head=dict(
        type='FCHead',
        in_channels=1,
        stride=8,
        scale_factor=1e-3,
        head_convs=None,
        loss=dict(type='BinaryCrossEntropyLoss', loss_weight=1.0),
        init_type='xavier_uniform'
    ),
)
test_cfg = dict(
    z_size=127,
    x_size=255,
    z_feat_size=6,
    x_feat_size=22,
    search_region=dict(num_scales=3, scale_step=1.0375, context_amount=0.5),
    upsampler=dict(scale_factor=16, mode='bicubic', align_corners=True),
    scale_penalty=0.9745,
    scale_damp=0.59,
    window=dict(weight=0.176),
    linear_update=dict(
        enable=False,
        init_portion=0.5,
        gamma=0.975
    )
)
train_cfg = dict(
    type='PairwiseWrapper',
    samples_per_gpu=8,
    workers_per_gpu=4,
    z_size=255,
    x_size=255,
    z_feat_size=6,
    x_feat_size=22,
    num_freeze_blocks=0,
    siamfc=dict(
        assigner=dict(type='CenterDistAssigner', pos_thresh=24, neg_thresh=30, dist_type='L1'),
        sampler=dict(type='PseudoSampler'),
        pos_weight=0.5,
    ),
    # optimizer
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    optimizer_config=dict(
        grad_clip=dict(max_norm=20, norm_type=2),
        optimizer_cfg=dict(
            type='SGD',
            params=[dict(name='all/weight'), dict(name='all/bias', weight_decay=0.0)],
            lr=0.01,
            weight_decay=0.0001
        ),
        optimizer_schedule=[]
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
