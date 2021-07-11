# predefined model ranges
stem_width_range = dict(
    key='arch.backbone.stem.width',
    start=32,
    end=64,
    step=16,
)
body_width_range = dict(
    key='arch.backbone.body.width',
    start=[48, 96, 192, 384],
    end=[80, 160, 320, 640],
    step=[16, 32, 64, 128],
    ascending=True,
)
body_depth_range = dict(
    key='arch.backbone.body.depth',
    start=[2, 2, 5, 2],
    end=[4, 6, 29, 4],
    step=[1, 2, 2, 1],
)

# predefined model anchors
MAX = {
    'name': 'MAX',
    'arch.backbone.stem.width': stem_width_range['end'],
    'arch.backbone.body.width': body_width_range['end'],
    'arch.backbone.body.depth': body_depth_range['end'],
}
MIN = {
    'name': 'MIN',
    'arch.backbone.stem.width': stem_width_range['start'],
    'arch.backbone.body.width': body_width_range['start'],
    'arch.backbone.body.depth': body_depth_range['start'],
}
R50 = {
    'name': 'R50',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 6, 3],
}
R77 = {
    'name': 'R77',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 15, 3],
}
R101 = {
    'name': 'R101',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3],
}

# config of model samplers
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
                dict(**MAX, ),
                dict(**MIN, ),
                dict(**R101, ),
                dict(**R77, ),
                dict(**R50, ),
            ]),
        # random model samplers
        dict(
            type='repeat',
            times=3,
            model_sampler=dict(
                type='composite',
                model_samplers=[
                    dict(
                        type='range',
                        **stem_width_range,
                    ),
                    dict(
                        type='range',
                        **body_width_range,
                    ),
                    dict(
                        type='range',
                        **body_depth_range,
                    ),
                ]))
    ])

val_sampler = dict(
    type='anchor', anchors=[
        dict(**R50, ),
        dict(**R77, ),
        dict(**R101, ),
    ])
