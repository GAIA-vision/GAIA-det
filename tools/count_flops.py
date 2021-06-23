# standard lib
import os
import sys
import os.path as osp
import argparse
import glob
import json
import shutil
from collections.abc import Sequence
from copy import deepcopy

# 3rd party lib
import torch
import torch.distributed as dist
from tqdm import tqdm

# mm lib
import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmdet.models import build_detector

# gaia lib
import gaiavision
import gaiadet
from gaiavision.utils import get_model_complexity_info
from gaiavision.model_space import build_model_sampler, fold_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Count flops of each subnet')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='train config file path')
    parser.add_argument(
        '--as_strings', action='store_true', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def prepare_cfg(cfg):
    # substitute dyn_bn for dyn_sync_bn
    model = cfg['model']
    for name in ('backbone', 'neck', 'rpn_head', 'roi_head'):
        m = model.get(name, None)
        if m is not None:
            if 'norm_cfg' in m.keys():
                m['norm_cfg'] = dict(type='DynBN')
    return cfg


def main():

    args = parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # abandon sync bn
    cfg = Config.fromfile(args.config)
    cfg = prepare_cfg(cfg)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # init distributed env first.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    tmpdir = osp.join(cfg.work_dir, '.rank_flops')
    if rank == 0:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=False)  # set `exist_ok=False` for safety

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # prepare model
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    # prepare model sampler
    model_sampler = build_model_sampler(cfg.train_sampler)
    model_sampler.set_mode('traverse')
    all_metas = list(model_sampler.traverse())
    metas_per_rank = all_metas[rank::world_size]

    metas = []
    if rank == 0:
        pbar = tqdm(total=len(metas_per_rank))
        interval = min(max(len(metas_per_rank) // 100, 1), 10)
    for n, model_meta in enumerate(metas_per_rank):
        model_meta = fold_dict(model_meta)
        arch_meta = model_meta['arch']
        data_meta = model_meta['data']
        model.manipulate_arch(arch_meta)
        input_shape = data_meta['input_shape']
        if not isinstance(input_shape, Sequence):
            input_shape = (3, input_shape, input_shape)
        elif isinstance(input_shape, str):  # 3,800,800
            input_shape = [int(v) for v in input_shape.strip().split(',')]
        flops, params = get_model_complexity_info(
            model,
            input_shape,
            print_per_layer_stat=False,
            as_strings=args.as_strings)

        overhead = {
            'flops': flops,
            'params': params,
        }
        meta = {
            'overhead': overhead,
            'arch': arch_meta,
            'data': data_meta,
        }
        metas.append(meta)
        if rank == 0:
            if n % interval == 0:
                pbar.update(interval)

    save_path = osp.join(tmpdir, f'flops.json.{rank}')
    with open(save_path, 'w') as fout:
        for meta in metas:
            fout.write(json.dumps(meta, ensure_ascii=False) + '\n')

    # merge all results
    dist.barrier()
    all_flops_fd = open(osp.join(cfg.work_dir, f'flops.json'), 'w')
    if rank == 0:
        for i in range(world_size):
            flop_path = osp.join(tmpdir, f'flops.json.{i}')
            with open(flop_path, 'r') as fin:
                for line_idx, line in enumerate(fin):
                    all_flops_fd.write(line)
        all_flops_fd.close()
        shutil.rmtree(tmpdir)
    print('Done.')


if __name__ == '__main__':
    main()
