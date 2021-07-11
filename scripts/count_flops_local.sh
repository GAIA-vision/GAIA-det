#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
PORT=${PORT:-29500}

# CONFIG=configs/local_examples/count_flops/faster_rcnn_ar50to101_flops.py

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/count_flops.py \
    ${CONFIG} \
    --launcher pytorch \
    --work-dir ${WORK_DIR}

