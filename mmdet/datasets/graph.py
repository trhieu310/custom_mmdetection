# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class GraphDataset(CocoDataset):

    # CLASSES = ('Table', 'Graphic', 'Geometry', 'Logo', 'Figure', 'Variation', 'Natural')
    CLASSES = ('Graphic', 'Natural', 'Variation', 'Figure', 'Logo', 'Geometry',
               'TB_Full_lined', 'TB_Merged_cells', 'TB_Partial_lined',
               'TB_Partial_line_MC', 'TB_No_lines')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30),]
