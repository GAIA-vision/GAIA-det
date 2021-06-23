# standard lib
import bisect
import math
from collections import defaultdict
import warnings
import pdb

# 3rd party lib
import numpy as np
import torch
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

# mm lib
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmcv.utils import print_log
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module()
class UniConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset with unified label space.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        label_pool (LabelPool): An instance of label pool which
            mapping of labels from separate datasets to unified
            label space.
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self,
                 datasets,
                 label_pool,
                 separate_eval=True,
                 test_mode=False,
                 manually_set_dataset_names=None):
        super(UniConcatDataset, self).__init__(datasets)
        self.label_pool = label_pool
        self.CLASSES = tuple(label_pool.names)
        self.separate_eval = separate_eval
        self.test_mode = test_mode

        # check order of included datasets
        def get_dataset_names(dataset):
            names = []
            if getattr(dataset, 'name', None):
                names.append(getattr(dataset, 'name'))
            # support exceptional dataset: CocoDataset
            elif isinstance(dataset, CocoDataset):
                names.append('coco')
                warnings.warn(
                    '`CocoDataset` applied, please check the `manually_set_dataset_names` '
                )
            elif getattr(dataset, 'dataset', None):
                names.extend(get_dataset_names(getattr(dataset, 'dataset')))
            elif getattr(dataset, 'datasets', None):
                for ds in getattr(dataset, 'datasets'):
                    names.extend(get_dataset_names(ds))
            return names

        if manually_set_dataset_names is None:
            self.dataset_names = get_dataset_names(self)
        else:
            self.dataset_names = manually_set_dataset_names

        dataset_pool = label_pool.dataset_names
        for dataset_name in self.dataset_names:
            assert dataset_name in dataset_pool, "dataset:{} name \
                must be included in label pool".format(dataset_name)
        assert len(datasets) == len(
            self.dataset_names
        ), f'{len(datasets)} vs. {len(self.dataset_names)}'

        if not separate_eval:
            if any([isinstance(ds, CocoDataset) for ds in datasets]):
                raise NotImplementedError(
                    'Evaluating concatenated CocoDataset as a whole is not'
                    ' supported! Please set "separate_eval=True"')
            elif len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        dataset_name = self.dataset_names[dataset_idx]
        label_mapping = self.label_pool.get_label_mapping(dataset_name)

        # process gt_labels if exist
        sample = self.datasets[dataset_idx][sample_idx]

        if not self.test_mode:
            sep_labels = sample['gt_labels'].data.tolist()
            uni_labels = [label_mapping.sep2uni(v) for v in sep_labels]
            sample['gt_labels'] = DC(to_tensor(uni_labels))

        if isinstance(sample['img_metas'], (list, tuple)):
            sample['img_metas'][0]._data['source'] = dataset_name
        else:
            sample['img_metas']._data['source'] = dataset_name

        return sample

    def get_cat_ids(self, idx):
        """ Get category ids of concatenated dataset by index.
            Note that the categories should be mapped to the
            unified label space.
        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        raw_cat_ids = self.datasets[dataset_idx].get_cat_ids(sample_idx)
        # remap separate labels to unified labels
        dataset_name = self.dataset_names[dataset_idx]
        label_mapping = self.label_pool.get_label_mapping(dataset_name)
        uni_cat_ids = [label_mapping.sep2uni(c) for c in raw_cat_ids]

        return uni_cat_ids

    def remap_label_space(self, results, dataset_name):
        label_mapping = self.label_pool.get_label_mapping(dataset_name)
        remapped_results = []
        for result_per_image in results:
            remapped_result_per_image = []
            for label in range(label_mapping.labels[-1] + 1):
                if label in label_mapping.labels:
                    uni_label = label_mapping.sep2uni(label)
                    remapped_result_per_image.append(
                        result_per_image[uni_label])
                else:
                    dummy_result = np.empty((0, 5), dtype=np.float32)
                    remapped_result_per_image.append(dummy_result)

            remapped_results.append(remapped_result_per_image)
        return remapped_results

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                    f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset, dataset_name in zip(self.cumulative_sizes,
                                                   self.datasets,
                                                   self.dataset_names):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                results_per_dataset = self.remap_label_space(
                    results_per_dataset, dataset_name)
                print_log(
                    f'\nEvaluateing {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_name}_{k}': v})

            return total_eval_results
        elif any([isinstance(ds, CocoDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Evaluating concatenated CocoDataset as a whole is not'
                ' supported! Please set "separate_eval=True"')
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results
