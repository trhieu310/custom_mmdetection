# dataset settings
classes = ('Illustration', 'Text', 'ScienceText')
dataset_type = 'DocslayoutDataset'
# data_root = 'data/coco/'
data_root = '../../input/doc-v28/'
img_norm_cfg = dict(
    mean=[239.99624306, 239.86340489, 240.44363462], std=[29.66910737, 29.56400222, 29.36436548], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=[
            data_root+'anns/v2.7.2.3_training.json', 
            data_root+'anns/v2.8.0_training.json', 
            data_root + 'anns/ann_r_0.0.0.json',
            data_root + 'anns/ann_r_1.0.1.json',
            data_root + 'anns/anns_dr_v1.0.0.json',
            data_root + 'anns/anns_r_1.0.2.json',
            data_root + 'anns/anns_r_1.0.3.json',
        ],
        img_prefix=data_root,
        seg_prefix=data_root + 'semantic_mask/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'anns/v2.8.0_testing.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'datatest/GT_fiftyone.json',
        img_prefix=data_root + 'datatest/images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
