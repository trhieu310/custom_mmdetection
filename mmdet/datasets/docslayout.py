from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DocslayoutDataset(CocoDataset):

    CLASSES = ('Illustration', 'Text', 'ScienceText')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142)]