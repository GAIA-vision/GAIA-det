model_space_path = 'path/to/ar50to101_flops.json'
model_sampling_rules = dict(
    type='sequential',
    rules=[
        # 1. overhead constraints, e.g. close to R50(138*1e9), replace it basing on your needs.
        dict(
            func_str=
            'lambda x: x[\'overhead.flops\'] <=140*1e9 and x[\'overhead.flops\'] >=135*1e9'
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
            ]),
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
            ]),
        # 4. sample 10 models within each group
        dict(
            type='sample',
            operation='random',
            value=10,
            mode='number',
        ),
        # 5. merge all groups
        dict(type='merge'),
    ])
