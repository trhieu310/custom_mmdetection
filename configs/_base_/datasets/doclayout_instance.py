# dataset settings
dataset_type = 'DocslayoutDataset'
data_root = '../../input/dcu272/'
json_root = '../../input/dcu272-json/'
img_norm_cfg = dict( 
    ###Hieunt - Change mean std for doclayouts
    mean=[236.56476823, 236.86395663, 237.62402599], std=[29.52565967, 29.79392858, 29.54714306], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0), ###Hieunt - set flip to 0.0
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32), ###Hieunt - Comment padding
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
            # dict(type='RandomFlip'), ###Hieunt - Comment flip
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32), ###Hieunt - Comment padding
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, ###Hieunt - change batchsize
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=json_root + 'train.json',  ###Hieunt - warninng update path 
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=json_root + 'valid.json', ###Hieunt - warninng update path 
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=json_root + 'test.json', ###Hieunt - warninng update path 
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
