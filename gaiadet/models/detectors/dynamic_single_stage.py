import pdb

# mm lib
from mmdet.models import DETECTORS, SingleStageDetector

# gaia lib
from gaiavision.core import DynamicMixin


@DETECTORS.register_module()
class DynamicSingleStageDetector(SingleStageDetector, DynamicMixin):
    search_space = {'backbone', 'neck', 'bbox_head'}

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DynamicSingleStageDetector, self).__init__(
                 backbone,
                 neck=neck,
                 bbox_head=bbox_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

    def manipulate_backbone(self, arch_meta):
        #pdb.set_trace()
        self.backbone.manipulate_arch(arch_meta)
        try:
            self.neck.manipulate_body(arch_meta['body']) # 这块似乎不太对劲？
        except:
            pdb.set_trace()

    def manipulate_neck(self, arch_meta):         
        #pdb.set_trace()
        try:
            self.neck.manipulate_arch(arch_meta)
        except:
            pdb.set_trace()
    def manipulate_bbox_head(self, arch_meta):
        try:
            self.bbox_head.manipulate_width(arch_meta['width'])
        except:
            pdb.set_trace()
        #raise NotImplementedError

