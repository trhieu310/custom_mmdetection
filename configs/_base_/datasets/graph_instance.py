# dataset settings
dataset_type = 'GraphDataset'
classes = ('Table', 'Graphic', 'Geometry', 'Logo', 'Figure', 'Variation', 'Natural')
# data_root = '/home/hieunt/Graph_Object_Data/'
data_root = '../../input/graphical/'
img_norm_cfg = dict(
    mean=[237.48618173, 236.966083, 237.09603447], std=[37.42294109, 37.52417463, 38.4937926], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='ResizeFreeLayout'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels' , 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='ResizeFreeLayout'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[data_root + 'anns/ann_part_29_graph.json',
                  data_root + 'anns/ann_part_24_graph.json',
                #   data_root + 'anns/ann_part_25_graph.json', 
                  data_root + 'anns/ann_part_26_graph.json'],
                #   data_root + 'anns/illustration.json',
                #   data_root + 'anns/table.json'],
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/ann_part_28_graph.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/ann_part_27_graph.json',
        # ann_file=data_root + 'anns/pr_v4.0.0_no_logo_table.json',
        img_prefix=data_root + 'images/', #pr_v4.0.0_graph/
        pipeline=test_pipeline))
evaluation = dict(classwise=True, metric=['bbox', 'segm'])
