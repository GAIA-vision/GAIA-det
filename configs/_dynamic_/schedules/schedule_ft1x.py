optimizer = dict(
    type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00001)  # 1e-4 -> 1e-5
optimizer_config = dict(grad_clip=None)
lr_scaler = dict(
    policy='linear',
    base_lr=0.0001875,  # 0.00125 -> 0.0001875
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    gamma=0.2,
    step=[9, 12])
total_epochs = 13
