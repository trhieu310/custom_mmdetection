# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class GraphDataset(CocoDataset):

    CLASSES = ('Table', 'Graphic', 'Geometry', 'Logo', 'Figure', 'Variation', 'Natural')

    PALETTE = [(0, 192, 64), (0, 64, 96), (128, 192, 192), (0, 64, 64),
               (0, 192, 224), (0, 192, 192), (128, 192, 64)]
