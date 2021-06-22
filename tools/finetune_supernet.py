# standard lib
import argparse
import copy
import os
import os.path as osp
import time
import warnings
from copy import deepcopy
import json
import pdb

# 3rd-parth lib
import torch
import torch.distributed as dist

# mm lib
import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.utils import get_git_hash
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet import __version__
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.utils import collect_env, get_root_logger
from mmdet.datasets import (build_dataloader, replace_ImageToTensor)

# gaia lib
import gaiavision
from gaiavision import broadcast_object
from gaiavision.model_space import (ModelSpaceManager, build_sample_rule,
                                    build_model_sampler, unfold_dict)

import gaiadet
from gaiadet.datasets import build_dataset
from gaiadet.apis import train_detector
from gaiadet.core import TestDistributedDataParallel
from gaiadet.datasets import ScaleManipulator, manipulate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a detector from supernet')
    parser.add_argument('config', help='finetune config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    # ------------------------- test arguments ---------------------------
    parser.add_argument('--load-from', help='cktp to be loaded')
    parser.add_argument(
        '--model-space-path',
        dest='model_space_path',
        default=None,
        help='model space path')
    parser.add_argument('--tmpdir', help='the tmp dir to store results')
    parser.add_argument(
        '--metric-tag',
        default='finetune',
        help='tag of metric in fast-ft process')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--out-name', default='metrics.json', help='output result file name')
    parser.add_argument(
        '--save-results', action='store_true', help='store results if true')
    # ------------------------- eval arguments ---------------------------
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    # ------------------------ routine arguments -------------------------
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if args.out is not None:
        warnings.warn(
            '`args.out` is suppressed here, the results are automatically stored in `args.work_dir/results/*.pkl`'
        )

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    # NOTE: It is extremely dangerous when `tmpdir` and `work_dir` are same.
    if args.tmpdir is not None:
        assert args.tmpdir.rstrip('/') != args.work_dir.rstrip('/')

    return args


def main():
    args = parse_args()

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

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.load_from is not None:
        cfg.load_from = args.load_from
        assert os.path.exists(cfg.load_from), f'`{cfg.load_from}` not existed.'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    # log some basic info
    logger.info(f'Distributed training: {distributed}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # CLI always first
    if args.model_space_path is not None:
        model_space_path = args.model_space_path
    else:
        model_space_path = cfg.model_space_path

    # in case of some stupid cloud service
    if args.model_space_path == 'None':
        model_space_path = cfg.model_space_path

    logger.info('Model space file loaded: {}'.format(model_space_path))

    # set up test dataset config
    test_samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    test_workers_per_gpu = cfg.data.workers_per_gpu
    if test_samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        try:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
        except AttributeError:
            if 'datasets' in cfg.data.test.keys():
                for ds in cfg.data.test.datasets:
                    ds.pipeline = replace_ImageToTensor(ds.pipeline)
            else:
                raise NotImplementedError

    # prepare model_space
    model_space = ModelSpaceManager.load(model_space_path)
    has_metric = False
    for c in model_space.columns:
        if 'metric' in c.strip():
            has_metric = True
            break
    if not has_metric:
        warnings.warn('`metric` is absent during finetuning.')

    # set up model sampling rules
    rule = build_sample_rule(cfg.model_sampling_rules)

    # apply rule
    sub_model_space = model_space.ms_manager.apply_rule(rule)
    model_metas = sub_model_space.ms_manager.pack()

    # store original cfg and meta
    ori_cfg = cfg
    ori_meta = meta
    fastft_model_metas = []

    for i, model_meta in enumerate(model_metas):
        # 0. clone cfg and meta, sync model metas
        cfg = deepcopy(ori_cfg)
        meta = deepcopy(ori_meta)
        model_meta = broadcast_object(model_meta)

        # 1. prepare dataset according to model_meta
        unfolded_model_meta = unfold_dict(model_meta)
        new_scale = unfolded_model_meta['data.input_shape'][-1]
        scale_manipulator = ScaleManipulator(new_scale)
        manipulate_dataset(cfg.data.train, scale_manipulator)
        manipulate_dataset(cfg.data.test, scale_manipulator)
        datasets = [build_dataset(cfg.data.train)]

        # 2. prepare train and val[dummy] model sampler
        model_sample_cfg = {
            'type': 'anchor',
            'anchors': [{
                'name': str(i),
                **unfolded_model_meta
            }],
        }
        train_sampler = build_model_sampler(model_sample_cfg)
        val_sampler = build_model_sampler(model_sample_cfg)

        # 3. train arch
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        train_detector(
            model,
            train_sampler,
            val_sampler,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

        # 4. manipulate arch and run test
        ckpt_path = osp.join(cfg.work_dir, 'latest.pth')
        ckpt = load_checkpoint(model, ckpt_path, map_location='cpu')

        model_for_test = TestDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        model_for_test.module.manipulate_arch(model_meta['arch'])
        anchor_name = model_meta.get('name',
                                     model_meta.get('index', 'n' + str(i)))

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        test_data_loader = build_dataloader(
            test_dataset,
            samples_per_gpu=test_samples_per_gpu,
            workers_per_gpu=test_workers_per_gpu,
            dist=distributed,
            shuffle=False)

        outputs = multi_gpu_test(model_for_test, test_data_loader, args.tmpdir,
                                 args.gpu_collect)
        res_model_meta = deepcopy(model_meta)

        # 5. evaluate the results
        if rank == 0:
            metrics = {}
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
            eval_res = test_dataset.evaluate(outputs, **eval_kwargs)

            for name, val in eval_res.items():
                metrics[name] = val

            logger.info('-- model meta:')
            logger.info(json.dumps(model_meta, indent=4))
            metric_meta = res_model_meta.setdefault('metric', {})
            metric_meta[args.metric_tag] = metrics
            res_model_meta['metric'] = metric_meta
            fastft_model_metas.append(res_model_meta)

        # 7. synchronize all procs
        dist.barrier()

    # 8. store all results. TODO: add checkpointing
    if rank == 0:
        metric_dir = args.work_dir
        os.makedirs(metric_dir, exist_ok=True)
        save_path = os.path.join(metric_dir, args.out_name)
        fastft_model_space = ModelSpaceManager.load(fastft_model_metas)
        fastft_model_space.ms_manager.dump(save_path)


if __name__ == '__main__':
    main()
