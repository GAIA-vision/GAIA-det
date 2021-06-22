import os.path as osp
import warnings
from math import inf
from collections.abc import Sequence

# 3rd parth lib
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# mm lib
import mmcv
from mmcv.runner import Hook

from mmdet.utils import get_root_logger
from mmdet.core import EvalHook

# gaia lib
from gaiavision import broadcast_object
from gaiavision import DynamicMixin, fold_dict

# local lib
from .test_parallel import TestDistributedDataParallel
from ...datasets import UniConcatDataset


class CrossArchEvalHook(EvalHook):
    """Evaluation hook on various model architectures.

    Notes:
        If new arguments are added for CrossEvalHook, tools/test.py,
        tools/eval_metric.py may be effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in best.json.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key will be used. The interval of
            ``CheckpointHook`` should device EvalHook. Default: None.
        rule (str, optional): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Keys such as 'mAP' or 'AR' will
            be inferred by 'greater' rule. Keys contain 'loss' will be inferred
             by 'less' rule. Options are 'greater', 'less'. Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['mAP', 'AR']
    less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 model_sampler,
                 start=None,
                 interval=1,
                 save_best=None,
                 key_indicator=None,
                 arch_name=None,
                 dataset_name=None,
                 rule=None,
                 **eval_kwargs):
        if isinstance(dataloader, DataLoader):
            dataloader = dataloader
        else:
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')

        if not interval > 0:
            raise ValueError(f'interval must be positive, but got {interval}')
        if start is not None and start < 0:
            warnings.warn(
                f'The evaluation start epoch {start} is smaller than 0, '
                f'use 0 instead', UserWarning)
            start = 0
        if not isinstance(dataloader.dataset, UniConcatDataset):
            assert dataset_name is None, (
                'Only support UniConcatDataset at the time')
        self.dataloader = dataloader
        self.model_sampler = model_sampler

        self.interval = interval
        self.start = start
        assert isinstance(save_best, str) or save_best is None
        self.save_best = save_best
        self.key_indicator = key_indicator
        self.arch_name = arch_name
        self.dataset_name = dataset_name

        self.eval_kwargs = eval_kwargs
        self.initial_epoch_flag = True

        self.logger = get_root_logger()

        if self.save_best is not None:
            self._init_rule(rule, self.save_best, self.key_indicator)

    def _init_rule(self, rule, save_best, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if save_best not in ('auto', 'data_avg', 'arch_avg', 'avg'):
            raise ValueError(f'`save_best` must be in (`auto`, `data_avg`, '
                             f'`arch_avg`, `avg`), but got `{save_best}`')

        if rule is None and key_indicator is not None:
            if any(key in key_indicator for key in self.greater_keys):
                rule = 'greater'
            elif any(key in key_indicator for key in self.less_keys):
                rule = 'less'
            else:
                raise ValueError(f'Cannot infer the rule for key '
                                 f'{key_indicator}, thus a specific rule '
                                 f'must be specified.')
        self.rule = rule
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def get_key_score(self, results, arch_name, dataset_name=None):
        if arch_name is None:
            if hasattr(self.model_sampler, 'anchor_name'):
                arch_name = self.model_sampler.anchor_name(0)
            else:
                arch_name = 0
        if dataset_name is None:
            dataset_name = self.dataloader.dataset.dataset_names[0]
        if self.key_indicator is None:
            self.key_indicator = '_'.join(
                list(results.keys())[0].split('_')[2:])

        # init rules
        self._init_rule(self.rule, self.save_best, self.key_indicator)

        if self.save_best == 'auto':
            key = f'{arch_name}_{dataset_name}_{self.key_indicator}'
            return results[key]

        vals = []
        if self.save_best == 'data_avg':
            for key, val in results.items():
                sp = key.split('_')
                metric = '_'.join(list(sp[2:]))
                if sp[0] == arch_name and metric == self.key_indicator:
                    vals.append(val)
        elif self.save_best == 'arch_avg':
            for key, val in results.items():
                sp = key.split('_')
                metric = '_'.join(list(sp[2:]))
                if sp[1] == dataset_name and metric == self.key_indicator:
                    vals.append(val)
        elif self.save_best == 'avg':
            for key, val in results.items():
                sp = key.split('_')
                metric = '_'.join(list(sp[2:]))
                if metric == self.key_indicator:
                    vals.append(val)
        else:
            raise NotImplementedError

        return sum(vals) / len(vals)

    def post_process_results(self, results, ds_name):
        results = self._remap_label_space(results, ds_name)
        return results

    def manipulate_arch(self, runner, arch_meta):
        if isinstance(runner.model, DynamicMixin):
            runner.model.manipulate_arch(arch_meta)
        elif isinstance(runner.model.module, DynamicMixin):
            runner.model.module.manipulate_arch(arch_meta)
        else:
            raise Exception(
                'Current model does not support arch manipulation.')

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return
        from mmdet.apis import single_gpu_test

        if not hasattr(self.model_sampler, 'traverse'):
            raise AttributeError(
                f'{type(self.model_sampler)} has no attribute `traverse`')
        all_res = {}
        for i, meta in enumerate(self.model_sampler.traverse()):
            if hasattr(self.model_sampler, 'anchor_name'):
                anchor_id = self.model_sampler.anchor_name(i)
            else:
                anchor_id = i

            meta = broadcast_object(fold_dict(meta))
            self.manipulate_arch(runner, meta['arch'])
            results = single_gpu_test(
                runner.model, self.dataloader, show=False)
            eval_res = self.evaluate(runner, self.dataloader, results)
            for name, val in eval_res.items():
                name = f'{anchor_id}_{name}'
                runner.log_buffer.output[name] = val
                all_res[name] = val
        runner.log_buffer.ready = True

        if self.save_best:
            key_score = self.get_key_score(all_res, self.arch_name,
                                           self.dataset_name)
            best_score = runner.meta['hook_msgs'].get(
                'best_score', self.init_value_map[self.rule])
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                last_ckpt = runner.meta['hook_msgs']['last_ckpt']
                runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
                mmcv.symlink(
                    last_ckpt,
                    osp.join(runner.work_dir,
                             f'best_{self.key_indicator}.pth'))
                self.logger.info(
                    f'Now best checkpoint is epoch_{runner.epoch + 1}.pth.'
                    f'Best {self.key_indicator} is {best_score:0.4f}')

    def evaluate(self, runner, dl, results):
        eval_res = dl.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        return eval_res


class DistCrossArchEvalHook(CrossArchEvalHook):
    """Distributed evaluation hook.

    Notes:
        If new arguments are added, tools/test.py may be effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be save in best.json.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
            segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
            ``auto``, the first key will be used. The interval of
            ``CheckpointHook`` should device EvalHook. Default: None.
        rule (str | None): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Default: 'None'.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(
            self,
            dataloader,
            model_sampler,
            start=None,
            interval=1,
            tmpdir=None,
            gpu_collect=True,  # for distributed cases
            save_best=None,
            key_indicator=None,
            arch_name=None,
            dataset_name=None,
            rule=None,
            **eval_kwargs):
        super().__init__(
            dataloader,
            model_sampler,
            start=start,
            interval=interval,
            save_best=save_best,
            key_indicator=key_indicator,
            arch_name=arch_name,
            dataset_name=dataset_name,
            rule=rule,
            **eval_kwargs)
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return

        from mmdet.apis import multi_gpu_test
        tmpdir = self.tmpdir

        if not hasattr(self.model_sampler, 'traverse'):
            raise AttributeError(
                f'{type(self.model_sampler)} has no attribute `traverse`')
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        if runner.rank == 0:
            all_res = {}

        model_for_test = TestDistributedDataParallel(
            runner.model.module.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        for i, meta in enumerate(self.model_sampler.traverse()):
            if hasattr(self.model_sampler, 'anchor_name'):
                anchor_id = self.model_sampler.anchor_name(i)
            else:
                anchor_id = i

            meta = broadcast_object(fold_dict(meta))
            self.manipulate_arch(runner, meta['arch'])
            results = multi_gpu_test(
                model_for_test,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect)
            if runner.rank == 0:
                print('\n')
                eval_res = self.evaluate(runner, self.dataloader, results)
                for name, val in eval_res.items():
                    name = f'{anchor_id}_{name}'
                    runner.log_buffer.output[name] = val
                    all_res[name] = val

        if runner.rank == 0:
            runner.log_buffer.ready = True
            if self.save_best:
                key_score = self.get_key_score(all_res, self.arch_name,
                                               self.dataset_name)
                best_score = runner.meta['hook_msgs'].get(
                    'best_score', self.init_value_map[self.rule])
                if self.compare_func(key_score, best_score):
                    best_score = key_score
                    runner.meta['hook_msgs']['best_score'] = best_score
                    last_ckpt = runner.meta['hook_msgs']['last_ckpt']
                    runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
                    mmcv.symlink(
                        last_ckpt,
                        osp.join(runner.work_dir,
                                 f'best_{self.key_indicator}.pth'))
                    self.logger.info(
                        f'Now best checkpoint is epoch_{runner.epoch + 1}.pth.'
                        f'Best {self.key_indicator} is {best_score:0.4f}')
