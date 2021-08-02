model_space_filename = 'path/to/metrics.json'

model_sampling_rules = dict(
    # NOTE: there are two strategies, pre-metric or post-metric, check which is better
    type='sequential',
    rules=[
        # 1. first 30% mAP models
        dict(
            type='sample',
            operation='top',
            # replace with customized metric in your own tasks, e.g. `metric.finetune.bdd100k_bbox_mAP`
            key='metric.direct.coco_bbox_mAP',
            value=0.3,
            mode='ratio',
        ),
        # 2. various scale constraints
        dict(
            type='parallel',
            rules=[
                dict(func_str='lambda x: x[\'data.input_shape\'] == 480'),
                dict(func_str='lambda x: x[\'data.input_shape\'] == 560'),
                dict(func_str='lambda x: x[\'data.input_shape\'] == 640'),
                dict(func_str='lambda x: x[\'data.input_shape\'] == 720'),
                dict(func_str='lambda x: x[\'data.input_shape\'] == 800'),
            ],
        ),
        # 3. various depth constraints
        dict(
            type='parallel',
            rules=[
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.backbone.body.depth\']) >= 11 and sum(x[\'arch.backbone.body.depth\']) < 17'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.backbone.body.depth\']) >= 18 and sum(x[\'arch.backbone.body.depth\']) < 24'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.backbone.body.depth\']) >= 25 and sum(x[\'arch.backbone.body.depth\']) < 31'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.backbone.body.depth\']) >= 32 and sum(x[\'arch.backbone.body.depth\']) < 38'
                ),
                dict(
                    func_str=
                    'lambda x: sum(x[\'arch.backbone.body.depth\']) >= 39 and sum(x[\'arch.backbone.body.depth\']) < 44'
                ),
            ]
        ),
        # 4. sample 5 models within each group
        dict(
            type='sample',
            operation='random',
            value=1, # 1 or 2 is acceptable for one node
            mode='number',
        ),
        # 5. merge all groups
        dict(type='merge'),
    ]
)
