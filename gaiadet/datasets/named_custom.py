# standard lib
import pdb
import math

# 3rd party lib
import torch
import numpy as np

# mm lib
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class NamedCustomDataset(CustomDataset):
    """ Named custom dataset for detection.

    There are several updated features compared to the CustomDataset.
    1) It owns a name;
    2) It supports filter_empty_gt.
    3) It filters all invalid gt-boxes, e.g. area < 1.
    4) It supports indexing only a protion of samples

    Args:
        name (str): Dataset name.
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
        sample_ratio (float, optional): sampling ratio.

    """

    def __init__(self,
                 name,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 sample_ratio=1.0):
        self.name = name
        self.sample_ratio = sample_ratio
        super(NamedCustomDataset,
              self).__init__(ann_file, pipeline, classes, data_root,
                             img_prefix, seg_prefix, proposal_file, test_mode,
                             filter_empty_gt)
        if sample_ratio < 1.0:
            g = torch.Generator()
            g.manual_seed(-1)
            self.shuffled_inds = torch.randperm(
                len(self.data_infos), generator=g).tolist()

        if test_mode:
            for i, img_info in enumerate(self.data_infos):
                if img_info.get('ann', None):
                    img_info['ann'] = self._parse_ann_info(
                        img_info, img_info['ann'])
                else:
                    bboxes = np.zeros((0, 4), dtype=np.float32)
                    labels = np.array([], dtype=np.int64)
                    img_info['ann'] = {'bboxes': bboxes, 'labels': labels}

    def __len__(self):
        return int(math.floor(self.sample_ratio * len(self.data_infos)))

    def __getitem__(self, idx):
        if self.sample_ratio < 1.0:
            idx = self.shuffled_inds[idx]

        if self.test_mode:
            return self.prepare_test_img(idx)

        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def shuffle_indexes(self, runner):
        g = torch.Generator()
        g.manual_seed(runner.epoch)
        self.shuffled_inds = torch.randperm(
            len(self.data_infos), generator=g).tolist()

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if img_info.get('ann', None) is None:
                continue

            img_info['ann'] = self._parse_ann_info(img_info, img_info['ann'])
            if self.filter_empty_gt:
                bboxes = img_info['ann'].get('bboxes', None)
                if bboxes is None:
                    continue
                elif len(bboxes) == 0:
                    continue

            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        img_h, img_w = img_info['height'], img_info['width']
        bboxes = ann_info['bboxes']
        bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if isinstance(bboxes, np.ndarray):
            raise NotImplementedError
        elif isinstance(bboxes, list):
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            for i, b in enumerate(bboxes):
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                inter_w = max(0, min(x2, img_w) - max(x1, 0))
                inter_h = max(0, min(y2, img_h) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if w < 1 or h < 1:
                    continue
                gt_bboxes.append(b)
                gt_labels.append(ann_info['labels'][i])

            if bboxes_ignore is not None:
                for i, b in enumerate(bboxes_ignore):
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    inter_w = max(0, min(x2, img_w) - max(x1, 0))
                    inter_h = max(0, min(y2, img_h) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if w < 1 or h < 1:
                        continue
                    gt_bboxes_ignore.append(b)
                    gt_labels_ignore.append(ann_info['labels_ignore'][i])

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

            if gt_bboxes_ignore:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            else:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
        )

        return ann
