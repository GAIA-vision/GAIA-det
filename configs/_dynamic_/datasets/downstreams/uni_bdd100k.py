# dataset types
dataset_type = 'CocoDataset'
data_root = '/path/to/bdd100k/'
# annotation root
CLASSES = ('person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic light', 'traffic sign', 'train')

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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ------------- dataset infomation -----------------
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.4.csv',
        dataset_names=['bdd100k'],
        datasets=[
            # for debug
            dict(
                type=dataset_type,
                classes=CLASSES,
                ann_file=data_root + 'labels/bdd100k_labels_images_det_coco_train.json',
                img_prefix=data_root+'images/100k/train/',
                pipeline=train_pipeline,
            ),
        ],
    ),
    val=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.4.csv',
        test_mode=True,
        dataset_names=['bdd100k'],
        datasets=[
            dict(
                type=dataset_type,
                classes=CLASSES,
                ann_file=data_root + 'labels/bdd100k_labels_images_det_coco_val.json',
                img_prefix=data_root + 'images/100k/val/',
                pipeline=test_pipeline
            ),
        ],
    ),
    test=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.4.csv',
        test_mode=True,
        dataset_names=['bdd100k'],
        datasets=[
            dict(
                type=dataset_type,
                classes=CLASSES,
                ann_file=data_root + 'labels/bdd100k_labels_images_det_coco_val.json',
                img_prefix=data_root + 'images/100k/val/',
                pipeline=test_pipeline
            ),
        ],
    ),
)
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='arch_avg',
    dataset_name='bdd100k',
)
