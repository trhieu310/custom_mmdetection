# dataset settings
import json
dataset_type = 'GraphDataset'
# classes = ('Table', 'Graphic', 'Geometry', 'Logo', 'Figure', 'Variation', 'Natural')
classes = ('Graphic', 'Natural', 'Variation', 'Figure', 'Logo', 'Geometry',
           'TB_Full_lined', 'TB_Merged_cells', 'TB_Partial_lined',
           'TB_Partial_line_MC', 'TB_No_lines')
# data_root = '/home/hieunt/Graph_Object_Data/'
# data_root = '../../input/graphical/'
data_root = '../../input/god-coco/'
test_root = "/home/null/GOD_DATA/pr_test/"
img_norm_cfg = dict(
    mean=[237.48618173, 236.966083, 237.09603447], std=[37.42294109, 37.52417463, 38.4937926], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='ResizeFreeLayout'),
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
            # dict(type='Resize', keep_ratio=True),
            dict(type='ResizeFreeLayout'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=[data_root + 'anns/R_1.0.2.json',
                  data_root + 'anns/R_1.0.3.json',
                  data_root + 'anns/Real_exam_1.json',
                  data_root + 'anns/Real_exam_2.json',
                  data_root + 'anns/Real_exam_3.json',
                  data_root + 'anns/Real_exam_4.json'],
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/ann_part_71.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'anns/ann_part_27_graph.json',

        # ann_file=data_root + 'true_pr_v4.0.0_graph.json',
        # img_prefix=data_root + 'pr_v4.0.0/pr_v4.0.0/',

        # img_prefix=data_root + 'images/', #pr_v4.0.0_graph/

        # ann_file='/home/null/custom_mmdetection/checkpoints/prv4/pr_v4.0.0_graph.json',
        # ann_file='/home/null/custom_mmdetection/checkpoints/prv4/graphical_pr_v4_0_0_inside_tb_seg.json',
        # ann_file='/home/null/custom_mmdetection/checkpoints/prv4/ann_crop.json',
        # img_prefix="/home/null/custom_mmdetection/checkpoints/prv4/",
        ann_file=test_root + 'anns/v4.0.0.json',
        img_prefix=test_root + "images/",
        # img_prefix="/home/null/custom_mmdetection/checkpoints/prv4/pr_v4.0.0/",
        pipeline=test_pipeline)
)
evaluation = dict(save_best='auto', classwise=True, metric=['bbox', 'segm'], iou_thrs=[0.5])  # iou_thrs=[0.5] -> set mAP


json.dumps
