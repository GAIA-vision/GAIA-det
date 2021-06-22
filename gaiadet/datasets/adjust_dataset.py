# standard lib
from collections.abc import Sequence

# mm lib
from mmcv.runner import HOOKS, Hook


class ScaleManipulator():

    def __init__(self, scale, offset=-1):
        self.scale = scale
        self.offset = offset

    def __call__(self, dataset_cfg):
        pipelines = dataset_cfg.pipeline
        for i, p in enumerate(pipelines):
            if p['type'] == 'Resize':
                # TODO: fix the bug during range
                ms_mode = p.get('multiscale_mode', 'range')
                img_scale = p['img_scale']
                if ms_mode == 'range':  # (max_size, short_edge)
                    if self.offset == -1:
                        offset = int((img_scale[1][1] - img_scale[0][1]) / 2)
                        p['img_scale'] = [
                            (img_scale[0][0], self.scale - offset),
                            (img_scale[1][0], self.scale + offset)
                        ]
                    else:
                        p['img_scale'] = [
                            (img_scale[0][0], self.scale - self.offset),
                            (img_scale[1][0], self.scale + self.offset)
                        ]
                elif ms_mode == 'value':  # (short_edge, max_size)
                    if isinstance(img_scale, tuple):
                        p['img_scale'] = (self.scale, img_scale[1])
                    elif isinstance(img_scale, list):
                        mean_scale = int(
                            sum([v[0] for v in img_scale]) / len(img_scale))
                        offset = self.scale - mean_scale
                        new_img_scale = [(v[0] + offset, v[1])
                                         for v in img_scale]
                        p['img_scale'] = new_img_scale

            elif p['type'] == 'MultiScaleFlipAug':
                img_scale = p['img_scale']
                if isinstance(img_scale, tuple):
                    p['img_scale'] = (img_scale[0], self.scale
                                      )  # (max_size, short_edge)
                elif isinstance(img_scale, list):
                    mean_scale = int(
                        sum([v[1] for v in img_scale]) / len(img_scale))
                    offset = self.scale - mean_scale
                    new_img_scale = [(v[0], v[1] + offset) for v in img_scale]
                    p['img_scale'] = new_img_scale


def manipulate_dataset(dataset, manipulator):
    if hasattr(dataset, 'dataset'):
        dataset = getattr(dataset, 'dataset')
        manipulate_dataset(dataset, manipulator)
    elif hasattr(dataset, 'datasets'):
        for dataset in getattr(dataset, 'datasets'):
            manipulate_dataset(dataset, manipulator)
    else:
        manipulator(dataset)


# @HOOKS.register_module()
# class AdjustDatasetHook(Hook):
#     def __init__(self, adjustments):
#         assert isinstance(adjustments, Sequence)
#         self.adjustments = adjustments
#
#     def before_train_epoch(self, runner):
#         dataset = runner.data_loader.dataset
#         for adjustment in self.adjustments:
#             adjustment = adjustment.copy()
#             adj_name = adjustment.pop('name')
#             if len(adjustment) > 0:
#                 AdjustDatasetHook.apply_adjustment(
#                     dataset, runner, adj_name, **adjustment)
#             else:
#                 AdjustDatasetHook.apply_adjustment(
#                     dataset, runner, adj_name)
#
#     @staticmethod
#     def apply_adjustment(dataset, runner, adjustment, **kwargs):
#         if hasattr(dataset, adjustment):
#             getattr(dataset, adjustment)(runner, **kwargs)
#
#         if hasattr(dataset, 'dataset'):
#             dataset = getattr(dataset, 'dataset')
#             AdjustDatasetHook.apply_adjustments(
#                 dataset, runner, adjustment, **kwargs)
#
#         if hasattr(dataset, 'datasets'):
#             for dataset in getattr(dataset, 'datasets'):
#                 AdjustDatasetHook.apply_adjustments(
#                     dataset, runner, adjustment, **kwargs)
