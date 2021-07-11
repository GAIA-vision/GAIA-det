_base_ = [
    '../../_dynamic_/models/faster_rcnn_fpn_ar50to101_gsync.py',
    '../../_dynamic_/datasets/uni_coco.py',
    '../../_dynamic_/schedules/schedule_ft2e.py',
    '../../_dynamic_/rules/ar50to101_ft2e_rules.py',
    '../../_base_/default_runtime.py',
]
fp16 = dict(loss_scale=512.)
train_cfg = dict(
    rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
    rcnn=dict(
        sampler=dict(
            _delete_=True,
            type='CombinedSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3))))
