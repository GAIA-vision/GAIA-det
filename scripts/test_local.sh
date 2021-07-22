#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
CKPT_PATH=$4
MODEL_SPACE_PATH=$5
PORT=${PORT:-29500}

# CONFIG=configs/local_examples/test_supernet/faster_rcnn_ar50to101_gsync_dist_eval.py

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/test_supernet.py \
    ${CONFIG} \
    ${CKPT_PATH} \
    --work-dir ${WORK_DIR} \
    --launcher pytorch \
    --eval bbox \
    --model-space-path ${MODEL_SPACE_PATH}
#     ${@:5}

