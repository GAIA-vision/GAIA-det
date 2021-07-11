model_space_filename = 'path/to/metrics.json'

model_sampling_rules = dict(
    type='sequential',
    rules=[
        # 1. select model with best performance, could replace with your own metrics
        dict(
            type='sample',
            operation='top',
            # replace with customized metric in your own tasks, e.g. `metric.finetune.bdd100k_bbox_mAP`
            key='metric.finetune.coco_bbox_mAP',
            value=1,
            mode='number',
        ),
    ])
