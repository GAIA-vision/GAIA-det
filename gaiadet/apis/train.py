# standard lib
import random
import math
import pdb

# 3rd-party lib
import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# mm lib
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.datasets import (build_dataloader, replace_ImageToTensor)
from mmdet.core import EvalHook, DistEvalHook
from mmdet.utils import get_root_logger
from mmdet.apis import set_random_seed

# gaia lib
from gaiavision.core import ManipulateArchHook

# local lib
from ..core.evaluation import CrossArchEvalHook, DistCrossArchEvalHook
from ..datasets import build_dataset, UniConcatDataset


def train_detector(
        model,
        train_sampler,  # model sampler
        val_sampler,  # model sampler
        dataset,
        cfg,
        distributed=False,
        validate=False,
        timestamp=None,
        meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # data_loader for training and validation
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        # Set `find_unused_parameters` True to enable sub-graph sampling
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        _, world_size = get_dist_info()
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        world_size = 1

    # decide whether to scale up lr
    lr_scaler_config = cfg.get('lr_scaler', None)
    if lr_scaler_config is not None:
        total_batch_size = world_size * cfg.data.samples_per_gpu
        base_lr = lr_scaler_config['base_lr']
        scale_type = lr_scaler_config.get('policy', 'linear')
        if scale_type == 'linear':
            scaled_lr = base_lr * total_batch_size
        elif scale_type == 'power':
            temp = lr_scaler_config.get('temperature', 0.7)
            scaled_lr = base_lr * math.pow(total_batch_size, temp)
        cfg.optimizer.lr = scaled_lr

    optimizer = build_optimizer(model, cfg.optimizer)
    # build runner
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # add hook for architecture manipulation
    manipulate_arch_hook = ManipulateArchHook(train_sampler)
    runner.register_hook(manipulate_arch_hook)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(val_dataset, UniConcatDataset):
            eval_hook = DistCrossArchEvalHook if distributed else CrossArchEvalHook
            runner.register_hook(
                eval_hook(val_dataloader, val_sampler, **eval_cfg))
        else:
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    calib_bn = cfg.get('caliberate_bn', None)
    if calib_bn:
        if calib_bn.get('reset_stats', False):

            def clean_bn_stats(m):
                if isinstance(m, _BatchNorm):
                    m.running_mean.zero_()
                    m.running_var.fill_(1)

            model.apply(clean_bn_stats)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
