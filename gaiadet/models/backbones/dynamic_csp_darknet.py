import pdb
import math

import torch
import torch.nn as nn

from gaiavision.core.bricks import DynamicConvModule, DynamicDepthwiseSeparableConvModule, DynamicSPPBottleneck, DynamicFocus
from gaiavision.core import DynamicMixin
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from ..utils import DynamicCSPLayer

@BACKBONES.register_module()
class DynamicCSPDarknet(BaseModule, DynamicMixin):
    search_space = {'stem', 'body'}

    def init_state(self, stem=None, body=None, **kwargs):
        # reserved state
        if stem is not None:
            self.stem_state = stem
        if body is not None:
            self.body_state = body

        for k, v in kwargs.items():
            setattr(self, f'{k}_state', v)


    def __init__(self,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='DynBN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DynamicDepthwiseSeparableConvModule if use_depthwise else DynamicConvModule

        self.stem = DynamicFocus(
            3,
            arch_setting[0][0],
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            num_blocks = max(num_blocks, 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = DynamicSPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            # spp layer的concat并不发生在最后，所以后面的csp layer 不用考虑concat的问题
            csp_layer = DynamicCSPLayer(
                out_channels,
                out_channels,
                depth=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.ModuleList(stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(DynamicCSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            import pdb
            #pdb.set_trace()
            if isinstance(layer,nn.ModuleList):
                #print("layer_name: ", layer_name, "layer depth： ", len(layer))
                for j in range(len(layer)):
                    x=layer[j](x)
            else:
                x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def manipulate_stem(self, arch_meta):
        #pdb.set_trace()
        self.stem_state = arch_meta
        self.stem.manipulate_arch(arch_meta)

    def manipulate_body(self, arch_meta):
        #pdb.set_trace()
        self.body_state = arch_meta
        # DL to LD
        sliced_arch_meta = [
            dict(zip(arch_meta, t)) for t in zip(*arch_meta.values())
        ]

        for i, layer_name in enumerate(self.layers):
            if layer_name == 'stem':
                continue
            stage_layer = getattr(self, layer_name)
            stage_layer[0].manipulate_width(sliced_arch_meta[i-1]['width'])
            stage_layer[-1].manipulate_width(sliced_arch_meta[i-1]['width'])
            stage_layer[-1].manipulate_depth(sliced_arch_meta[i-1]['depth'])
            if len(stage_layer) == 3: #说明用了DynamicSPPBottleneck
                stage_layer[1].manipulate_width(sliced_arch_meta[i-1]['width'])

            '''
            if layer_name!='stem':
                stage_layer = getattr(self, layer_name)

                if arch_meta=='width':# 问题好像出在了body的manipulate，arch_meta就没有是width的时候。。。
                                      # 这一行似乎永远不会被执行
                    assert 1==2
                    stage_layer[0].manipulate_width(sliced_arch_meta[i-1][arch_meta])
                    if len(stage_layer)>2:
                        stage_layer[1].manipulate_width1(sliced_arch_meta[i-1][arch_meta]//2)
                    for j in range(len(stage_layer)-1):
                        stage_layer[j+1].manipulate_arch(sliced_arch_meta[i-1])
                else:
                    stage_layer[-1].manipulate_arch(sliced_arch_meta[i-1])
            '''