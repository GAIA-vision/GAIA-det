model_space_filename = '/path/to/metrics.json'

model_sampling_rules = dict(
    # NOTE: there are two strategies, pre-metric or post-metric, check which is better
    type='sequential',
    rules=[
        # 1. first 50% mAP models
        dict(
            type='sample',
            operation='top',
            key='metric.coco_bbox_mAP',
            value=0.5,
            mode='ratio',
        ),
        # 2. various scale constraints
        dict(
            type='parallel',
            rules=[
                dict(func_str='lambda x: x[\'input_shape\'][-1] == 480'),
                dict(func_str='lambda x: x[\'input_shape\'][-1] == 560'),
                dict(func_str='lambda x: x[\'input_shape\'][-1] == 640'),
                dict(func_str='lambda x: x[\'input_shape\'][-1] == 720'),
                dict(func_str='lambda x: x[\'input_shape\'][-1] == 800'),
            ],
        ),
        # 3. various depth constraints
        dict(
            type='parallel',
            rules=[
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.body_depths\']) >= 11 and sum(x[\'arch.body_depths\']) < 17'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.body_depths\']) >= 18 and sum(x[\'arch.body_depths\']) < 24'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.body_depths\']) >= 25 and sum(x[\'arch.body_depths\']) < 31'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.body_depths\']) >= 32 and sum(x[\'arch.body_depths\']) < 38'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.body_depths\']) >= 39 and sum(x[\'arch.body_depths\']) < 44'
                ),
            ],
        ),
        # 4. sample 2 models within each group
        dict(
            type='sample',
            operation='random',
            value=2,
            mode='number',
        ),
        # 5. merge all groups
        dict(type='merge'),
    ])
