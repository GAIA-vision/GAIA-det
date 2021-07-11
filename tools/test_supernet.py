# standard lib
import argparse
import os
import os.path as osp
import time
import json
import warnings
from copy import deepcopy
import pdb

# 3rd-party lib
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm

# mm lib
import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import get_root_logger

# gaia lib
import gaiavision
from gaiavision import broadcast_object
from gaiavision.model_space import ModelSpaceManager, build_sample_rule, build_model_sampler

import gaiadet
from gaiadet.datasets import build_dataset
from gaiadet.apis import train_detector
from gaiadet.core import TestDistributedDataParallel
from gaiadet.datasets import ScaleManipulator, manipulate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='GAIA test (and eval) many models')
    parser.add_argument('config', help='test config file path')
    # ------------------------- test arguments ---------------------------
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--model-space-path', dest='model_space_path', help='model space path')
    parser.add_argument('--work-dir', help='the dir to save metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--out-name', default='metrics.json', help='output result file name')
    parser.add_argument(
        '--save-results', action='store_true', help='store results if true')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    # ------------------------- eval arguments ---------------------------
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--metric-tag',
        default='direct',
        help='tag of metric in fast-ft process')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    metric_dir = os.path.join(args.work_dir, 'test_supernet')
    args.metric_dir = metric_dir

    if args.out is not None:
        warnings.warn(
            '`args.out` is suppressed here, the results are automatically stored in `args.work_dir/results/*.pkl`'
        )

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # NOTE: it is extremely dangerous when `tmpdir` and `work_dir` are same.
    # an unforgettably terrible experience...
    if args.tmpdir is not None:
        assert args.tmpdir.rstrip('/') != args.work_dir.rstrip('/')

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    rank, _ = get_dist_info()

    # init the logger before other steps
    os.makedirs(args.metric_dir, exist_ok=True)
    save_path = os.path.join(args.metric_dir, args.out_name)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.metric_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        try:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
        except AttributeError:
            # no need to do the resursion, since the dataset is simple during inference
            if 'datasets' in cfg.data.test.keys():
                for ds in cfg.data.test.datasets:
                    ds.pipeline = replace_ImageToTensor(ds.pipeline)
            else:
                raise NotImplementedError

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    calib_bn = cfg.get('caliberate_bn', None)
    if calib_bn:
        if calib_bn.get('use_minibatch_stats', False):

            def clean_bn_stats(m):
                if isinstance(m, _BatchNorm):
                    m.running_mean = None
                    m.running_var = None
                    m.track_running_stats = False

            model.apply(clean_bn_stats)

    # set up model sampling rules
    rule = build_sample_rule(cfg.model_sampling_rules)

    # build up model space
    if args.model_space_path:
        model_space_path = args.model_space_path
    else:
        model_space_path = cfg.model_space_path
    model_space = ModelSpaceManager.load(model_space_path)
    sub_model_space = model_space.ms_manager.apply_rule(rule)
    model_metas = sub_model_space.ms_manager.pack()

    # TODO: fix the problem of fusing dynamic conv_bn
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)

    sampled_model_metas = []
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        for i, model_meta in enumerate(model_metas):
            # manipulate data
            new_scale = model_meta['data']['input_shape'][-1]
            scale_manipulator = ScaleManipulator(new_scale)
            manipulate_dataset(cfg.data.test, scale_manipulator)

            dataset = build_dataset(cfg.data.test, dict(test_mode=True))
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES

            # manipulate arch
            model.module.manipulate_arch(model_meta['arch'])
            anchor_name = model_meta.get('name', 'm' + str(i))

            # run test
            outputs = single_gpu_test(model, data_loader, args.show,
                                      args.show_dir, args.show_score_thr)
            result_model_meta = deepcopy(model_meta)
            metrics = {}
            print(f'\nwriting results to {args.out}')
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            eval_res = dataset.evaluate(outputs, **eval_kwargs)
            for name, val in eval_res.items():
                metrics[name] = val

            result_model_meta['metric'] = metrics
            sampled_model_metas.append(result_model_meta)
    else:
        model = TestDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        for i, model_meta in enumerate(model_metas):
            # sync model_meta between ranks
            model_meta = broadcast_object(model_meta)

            # manipulate data
            new_scale = model_meta['data']['input_shape'][-1]
            scale_manipulator = ScaleManipulator(new_scale)
            manipulate_dataset(cfg.data.test, scale_manipulator)

            dataset = build_dataset(cfg.data.test, dict(test_mode=True))
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

            # manipulate arch
            model.module.manipulate_arch(model_meta['arch'])
            anchor_name = model_meta.get('name', 'm' + str(i))

            # run test
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
            result_model_meta = deepcopy(model_meta)
            metrics = {}
            if rank == 0:
                if args.save_results:
                    result_dir = osp.join(args.work_dir, 'results')
                    os.makedirs(result_dir, exist_ok=True)
                    result_path = osp.join(result_dir, f'{anchor_name}.pkl')
                    print(f'\nwriting results to {result_path}')
                    mmcv.dump(outputs, result_path)

                eval_kwargs = cfg.get('evaluation', {}).copy()
                kwargs = {} if args.eval_options is None else args.eval_options
                for key in [
                        'interval',
                        'tmpdir',
                        'start',
                        'gpu_collect',
                        'save_best',
                        'rule',
                        'dataset_name',
                        'arch_name',
                ]:
                    eval_kwargs.pop(key, None)

                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                eval_res = dataset.evaluate(outputs, **eval_kwargs)

                for name, val in eval_res.items():
                    metrics[name] = val

                metric_meta = result_model_meta.setdefault('metric', {})
                metric_meta[args.metric_tag] = metrics
                result_model_meta['metric'] = metric_meta
                sampled_model_metas.append(result_model_meta)
                logger.info('-- model meta:')
                logger.info(json.dumps(result_model_meta, indent=4))
        dist.barrier()

    if rank == 0:
        sub_model_space = ModelSpaceManager.load(sampled_model_metas)
        sub_model_space.ms_manager.dump(save_path)


if __name__ == '__main__':
    main()
