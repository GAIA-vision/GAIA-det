# standard library
import warnings

# 3rd party library
from torch import nn as nn

# mm library
from mmcv.cnn import build_conv_layer

# gaia lib
from gaiavision.core import DynamicMixin
from gaiavision.core.bricks import build_norm_layer, DynamicBottleneck


class DynamicResLayer(nn.ModuleList, DynamicMixin):
    """DynamicResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build DynamicResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        depth (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """
    search_space = {'depth', 'width'}

    def init_state(self, depth=None, width=None, **kwargs):
        # reserved state
        if depth is not None:
            self.depth_state = depth
        if width is not None:
            self.width_state = depth

        for k, v in kwargs.items():
            setattr(self, f'{k}_state', v)

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 depth,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 downsample_first=True,
                 **kwargs):
        # TODO: fix the workaround
        if conv_cfg['type'] != 'DynConv2d':
            warnings.warn('Non-dynamic-conv detected in dynamic block.')
        if 'Dyn' not in norm_cfg['type']:
            warnings.warn('Non-dynamic-bn detected in dynamic block.')

        self.block = block
        self.avg_down = avg_down
        # TODO: support other states
        self.init_state(depth=depth, width=planes)

        # build downsample branch
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    padding=0,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, depth):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(depth - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(DynamicResLayer, self).__init__(layers)

    def manipulate_depth(self, depth):
        assert depth >= 1, 'Depth must be greater than 0, ' \
                           'skipping stage is not supported yet.'
        self.depth_state = depth

    def manipulate_width(self, width):
        self.width_stage = width
        for m in self:
            m.manipulate_width(width)

    def deploy_forward(self, x):
        # remove unused layers based on depth_state
        del self[self.depth_state:]
        for i in range(self.depth_state):
            x = self[i](x)
        return x

    def forward(self, x):
        if getattr(self, '_deploying', False):
            return self.deploy_forward(x)

        for i in range(self.depth_state):
            x = self[i](x)
        return x
