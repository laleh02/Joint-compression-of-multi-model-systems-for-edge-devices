optimizer = dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=0)
optimizer_config = dict(grad_clip=None)

lr_mult = 8
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[50 * lr_mult, 68 * lr_mult])
runner = dict(type='EpochBasedRunner', max_epochs=1)

checkpoint_config = dict(interval=80)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RetinaFaceDataset'
data_root = 'logynthetic/train/'
train_root = 'logynthetic/train/'
val_root = 'logynthetic/calibration/'
img_norm_cfg = dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=False)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RetinaFaceDataset',
        ann_file='logynthetic/yunet_labels/pipelined_train_annotations.txt',
        img_prefix='logynthetic/train/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            dict(
                type='RandomSquareCrop',
                crop_choice=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_keypointss'
                ])
        ]),
    val=dict(
        type='RetinaFaceDataset',
        ann_file='logynthetic/yunet_labels/pipelined_calibration_annotations.txt',
        img_prefix='logynthetic/calibration/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.],
                        to_rgb=False),
                    dict(type='Pad', size=(256, 256), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RetinaFaceDataset',
        ann_file='logynthetic/yunet_labels/pipelined_test_annotations.txt',
        img_prefix='logynthetic/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.],
                        to_rgb=False),
                    dict(type='Pad', size=(256, 256), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

model = dict(
    type='PipelinedYuNet',
    backbone=dict(
        type='YuNetBackbone',
        stage_channels=[[3, 16, 16], [16, 64], [64, 64], [64, 64], [64, 64],
                        [64, 64]],
        downsample_idx=[0, 2, 3, 4],
        out_idx=[3, 4, 5]),
    neck=dict(type='TFPN', in_channels=[64, 64, 64], out_idx=[0, 1, 2]),
    bbox_head=dict(
        type='YuNet_Head',
        num_classes=1,
        in_channels=64,
        shared_stacked_convs=1,
        stacked_convs=0,
        feat_channels=64,
        prior_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(type='EIoULoss', loss_weight=5.0, reduction='sum'),
        use_kps=True,
        kps_num=5,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(
        nms_pre=-1,
        min_bbox_size=0,
        score_thr=0.02,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=-1,
    ))
evaluation = dict(interval=1001, metric='mAP')
