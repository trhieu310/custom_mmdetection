model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=11,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'GraphDataset'
classes = ('Graphic', 'Natural', 'Variation', 'Figure', 'Logo', 'Geometry',
           'TB_Full_lined', 'TB_Merged_cells', 'TB_Partial_lined',
           'TB_Partial_line_MC', 'TB_No_lines')
data_root = '../../../../HDD/hieunt/datasets/GOD_V1.3.0/'
test_root = "/home/null/GOD_DATA/pr_test/"
img_norm_cfg = dict(
    mean=[237.48618173, 236.966083, 237.09603447],
    std=[37.42294109, 37.52417463, 38.4937926],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='ResizeFreeLayout'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[237.48618173, 236.966083, 237.09603447],
        std=[37.42294109, 37.52417463, 38.4937926],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='ResizeFreeLayout'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[237.48618173, 236.966083, 237.09603447],
                std=[37.42294109, 37.52417463, 38.4937926],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='GraphDataset',
        ann_file=[
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/r_1.0.2.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/r_1.0.3.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_60.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_62.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_63.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_64.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_65.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_66.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_67.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_68.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/lite_ann_part_69.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_70.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/real_exam_1.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/real_exam_2.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/real_exam_4.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/r_1.0.2_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_60_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_62_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_63_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_64_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_65_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_66_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_67_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_68_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_69_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_part_70_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_real_exam_1_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_real_exam_2_crop.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_real_exam_4_crop.json'
        ],
        img_prefix='../../../../HDD/hieunt/datasets/GOD_V1.3.0/images/',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='ResizeFreeLayout'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[237.48618173, 236.966083, 237.09603447],
                std=[37.42294109, 37.52417463, 38.4937926],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='GraphDataset',
        ann_file=[
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/real_exam_3.json',
            '../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/ann_real_exam_3_crop.json'
        ],
        img_prefix='../../../../HDD/hieunt/datasets/GOD_V1.3.0/images/',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='ResizeFreeLayout'),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[237.48618173, 236.966083, 237.09603447],
                        std=[37.42294109, 37.52417463, 38.4937926],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='GraphDataset',
        # ann_file='../../../../HDD/hieunt/datasets/GOD_V1.3.0/anns/pr_v4.0.0_no_logo_table.json',
        # img_prefix='../../../../HDD/hieunt/datasets/GOD_V1.3.0/images/pr_v4.0.0_graph/',
        # ann_V4.0.0_crop.json, ann_V4.1.0_crop.json, ann_V5.0.0_crop.json, v4.0.0.json, v4.1.0.json, v5.0.0.json
        ann_file=test_root + 'anns/ann_V4.1.0_crop.json',
        img_prefix=test_root + "images/",
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='ResizeFreeLayout'),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[237.48618173, 236.966083, 237.09603447],
                        std=[37.42294109, 37.52417463, 38.4937926],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(classwise=True, metric=['bbox'])
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=280)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/epoch_137.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=32)
work_dir = './work_dirs/cascade_mask_rcnn_r50_fpn_1x_graph'
auto_resume = False
gpu_ids = [0]
