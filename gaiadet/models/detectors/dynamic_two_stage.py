# mm lib
from mmdet.models import DETECTORS, TwoStageDetector

# gaia lib
from gaiavision.core import DynamicMixin


@DETECTORS.register_module()
class DynamicTwoStageDetector(TwoStageDetector, DynamicMixin):
    search_space = {'backbone', 'neck', 'roi_head', 'rpn_head'}

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(DynamicTwoStageDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)

    def manipulate_neck(self, arch_meta):
        raise NotImplementedError

    def manipulate_rpn_head(self, arch_meta):
        raise NotImplementedError

    def manipulate_roi_head(self, arch_meta):
        raise NotImplementedError
