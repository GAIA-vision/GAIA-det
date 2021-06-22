# standard lib
import logging

# 3rd-party lib
import torch
import torch.distributed as dist

# mm lib
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.parallel.distributed import MMDistributedDataParallel


class TestDistributedDataParallel(MMDistributedDataParallel):
    '''
    Adapt MMDistributedDataParallel with torch >= 1.8.0
    '''

    def to_kwargs(self, inputs, kwargs, device_id):
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)
