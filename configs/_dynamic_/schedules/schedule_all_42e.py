# optimizer
optimizer = dict(
    type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-5)  # default 1e-4
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2), )
# learning policy
lr_scaler = dict(
    policy='linear',
    base_lr=0.00125, # real learning rate per image
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    step=[32, 38, 41])
total_epochs = 42
