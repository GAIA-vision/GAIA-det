optimizer = dict(
    type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00001)  # default 1e-4 -> 1e-5
optimizer_config = dict(grad_clip=None)
lr_scaler = dict(
    policy='linear',
    base_lr=0.0001875,  # 0.00125 -> 0.0001875
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_by_epoch=False,
    step=[1])
total_epochs = 2
