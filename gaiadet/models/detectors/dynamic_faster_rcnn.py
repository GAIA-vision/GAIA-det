# mm lib
from mmdet.models import DETECTORS

# gaia lib
from gaiavision.core import DynamicMixin

# local lib
from .dynamic_two_stage import DynamicTwoStageDetector


@DETECTORS.register_module()
class DynamicFasterRCNN(DynamicTwoStageDetector, DynamicMixin):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(DynamicFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
