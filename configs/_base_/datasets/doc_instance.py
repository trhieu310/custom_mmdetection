# dataset settings
dataset_type = 'DocslayoutDataset'
# data_root = '../../../../../../../mnt/datadrive2/quangdq/v2.7.2/'
data_root = '../../input/cascade-mask-rcnn-data/'
img_norm_cfg = dict(
    mean=[236.56476823, 236.86395663, 237.62402599], std=[29.52565967, 29.79392858, 29.54714306], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'whole_training_2.7.2.2.json',
        img_prefix=data_root + 'doc_cascade/doc_data',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'testing.json',
        img_prefix=data_root + 'doc_cascade/doc_data',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'testing.json',
        img_prefix=data_root + 'doc_cascade/doc_data',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
