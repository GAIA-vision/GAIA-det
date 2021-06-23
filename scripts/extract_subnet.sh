#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
SRC_CKPT=$4
PORT=${PORT:-29500}

# CONFIG=configs/local_examples/extract_subnet/faster_rcnn_ar50_extract.py

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/extract_subnet.py \
    ${SRC_CKPT} \
    ${WORK_DIR} \
    ${CONFIG} \
    --launcher pytorch

