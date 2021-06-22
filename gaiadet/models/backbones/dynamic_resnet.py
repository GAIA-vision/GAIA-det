# standard lib
import pdb
from collections.abc import Sequence

# 3rd party lib
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

# mm lib
from mmcv.cnn import (build_plugin_layer, constant_init, kaiming_init,
                      build_conv_layer, build_activation_layer)
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES

# gaia lib
from gaiavision.core import DynamicMixin
from gaiavision.core.bricks import build_norm_layer, DynamicBottleneck

# local lib
from ..utils import DynamicResLayer


@BACKBONES.register_module()
class DynamicResNet(nn.Module, DynamicMixin):
    """DynamicResNet backbone.

    Args:
        stem_width (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
        frozen_stages (list): Number of layers in each stage to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config activation layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        pass
    """
    search_space = {'stem', 'body'}

    def init_state(self, stem=None, body=None, **kwargs):
        # reserved state
        if stem is not None:
            self.stem_state = stem
        if body is not None:
            self.body_state = body

        for k, v in kwargs.items():
            setattr(self, f'{k}_state', v)

    def __init__(
            self,
            in_channels,
            stem_width,
            body_width,  # width of each stage
            body_depth,  # depth of each stage
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1, 2, 3),
            style='pytorch',
            deep_stem=False,
            avg_down=False,
            frozen_stages=-1,
            frozen_layers=None,
            conv_cfg=None,
            norm_cfg=dict(type='DynSyncBN'),
            act_cfg=dict(type='ReLU'),
            norm_eval=False,
            dcn=None,
            stage_with_dcn=(False, False, False, False),
            plugins=None,
            with_cp=False,
            zero_init_residual=True):
        super(DynamicResNet, self).__init__()
        self.body_depth = body_depth
        self.stem_width = stem_width
        self.body_width = body_width
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.frozen_layers = frozen_layers
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        # TODO: support DynamicBasicBlock
        self.block = DynamicBottleneck
        self.body_depth = body_depth[:num_stages]
        self.inplanes = stem_width

        self.init_state(
            stem={'width': stem_width},
            body={
                'depth': body_depth,
                'width': body_width
            })

        self._make_stem_layer(in_channels, stem_width)

        self.res_layers = []
        for i, num_blocks in enumerate(self.body_depth):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = body_width[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                depth=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # freeze selected stages
        self._freeze_stages()
        # freeze selected layers in each stages
        self._freeze_layers()

        self.feat_dim = self.block.expansion * body_width[0] * \
            2**(len(self.body_depth) - 1)
        self.active_feat_dim = self.feat_dim

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for DynResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = DynamicResNet(...)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return DynamicResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_width):
        if self.deep_stem:
            assert isinstance(stem_width, Sequence)
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_width[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_width[0])[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_width[0],
                    stem_width[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_width[1])[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_width[1],
                    stem_width[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_width[2])[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_width,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_width, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _freeze_layers(self):
        """Freeze the first n layers in each stages
        """
        if self.frozen_layers is not None:
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                frozen_layer_num = self.frozen_layers[i]
                assert frozen_layer_num <= len(res_layer)
                for j in range(frozen_layer_num):
                    m = res_layer[j]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, DynamicBottleneck):
                        constant_init(m.norm3, 0)
                    # elif isinstance(m, DynamicBasicBlock):
                    #     constant_init(m.norm2, 0)

        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DynamicResNet, self).train(mode)
        self._freeze_stages()
        self._freeze_layers()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def manipulate_stem(self, arch_meta):
        """stem_width is a stom search space.
        Example:
        arch_meta = {'width': 32} or
        arch_meta = {'width': [16, 16, 32]} for deep_stem
        """
        self.stem_state = arch_meta
        if self.deep_stem:
            # DL to LD
            sliced_arch_meta = [
                dict(zip(arch_meta, t)) for t in zip(*arch_meta.values())
            ]
            self.stem[0].manipulate_arch(sliced_arch_meta[0])  # conv, bn, act
            self.stem[3].manipulate_arch(sliced_arch_meta[1])  # conv, bn, act
            self.stem[6].manipulate_arch(sliced_arch_meta[2])  # conv, bn, act
        else:
            self.conv1.manipulate_arch(arch_meta)

    def manipulate_body(self, arch_meta):
        self.body_state = arch_meta
        # DL to LD
        sliced_arch_meta = [
            dict(zip(arch_meta, t)) for t in zip(*arch_meta.values())
        ]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            res_layer.manipulate_arch(sliced_arch_meta[i])

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
