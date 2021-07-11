input_shape_cands = dict(
    key='data.input_shape', candidates=(480, 560, 640, 720, 800, 880, 960))
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
    'data.input_shape': 800,
}
MIN = {
    'name': 'MIN',
    'arch.backbone.stem.width': stem_width_range['start'],
    'arch.backbone.body.width': body_width_range['start'],
    'arch.backbone.body.depth': body_depth_range['start'],
    'data.input_shape': 800,
}
R50 = {
    'name': 'R50',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 6, 3],
    'data.input_shape': 800,
}
R77 = {
    'name': 'R77',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 15, 3],
    'data.input_shape': 800,
}
R101 = {
    'name': 'R101',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3],
    'data.input_shape': 800,
}

train_sampler = dict(
    type='anchor', anchors=[
        dict(**R50, ),
    ])
