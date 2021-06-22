from mmdet.models import DETECTORS, TwoStageDetector


@DETECTORS.register_module()
class DynamicTwoStageDetector(TwoStageDetector):

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

    def _manipulate_backbone_arch(self, backbone_cfg):
        assert hasattr(self.backbone, 'manipulate_arch'), \
            f'Backbone({type(self.backnone)}) has no method named `manipulate_arch`'
        self.backbone.manipulate_arch(**backbone_cfg)
        if hasattr(self.backbone, 'arch_state'):
            return self.backbone.arch_state()
        else:
            return None

    def _manipulate_neck_arch(self, arch_state, cfg=None):
        raise NotImplementedError

    def _manipulate_rpn_head_arch(self, arch_state, cfg=None):
        raise NotImplementedError

    def _manipulate_roi_head_arch(self, arch_state, cfg=None):
        raise NotImplementedError

    def manipulate_arch(self, backbone_cfg=None, neck_cfg=None, head_cfg=None):
        # TODO: support arch manipulation in neck and head
        backbone_state = self._manipulate_backbone_arch(backbone_cfg)
