# dataset types
dataset_types = {
    'coco': 'CocoDataset',
    'object365': 'NamedCustomDataset',
    'openimages': 'NamedCustomDataset',
}

# data root
data_roots = {
    'coco': '/path/to/coco',
    'cocotrain': '/path/to/coco/images/train2017',
    'cocoval': '/path/to/coco/images/val2017',
    'openimages': '/path/to/openimages',
    'object365': '/path/to/object365',
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
        multiscale_mode='range',
        img_scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ------------- dataset infomation -----------------
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        datasets=[
            # coco
            dict(
                type=dataset_types['coco'],
                ann_file=data_roots['coco'] + '/annotations/instances_train2017.json',
                img_prefix=data_roots['coco'] + '/images/train2017',
                pipeline=train_pipeline,
            ),
        ],
    ),
    val=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        dataset_names=['coco'],
        test_mode=True,
        datasets=[
            dict(
                type=dataset_types['coco'],
                ann_file=data_roots['coco'] + '/annotations/instances_val2017.json',
                img_prefix=data_roots['coco'] + '/images/val2017',
                pipeline=test_pipeline),
        ],
    ),
    test=dict(
        samples_per_gpu=8,
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        dataset_names=['coco'],
        test_mode=True,
        datasets=[
            dict(
                type=dataset_types['coco'],
                ann_file=data_roots['coco'] + '/annotations/instances_val2017.json',
                img_prefix=data_roots['coco'] + '/images/val2017',
                pipeline=test_pipeline),
        ],
    ),
)
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='arch_avg',
    dataset_name='coco',
)
