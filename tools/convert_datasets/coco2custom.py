# standard lib
import os
from pathlib import Path
import argparse
import json
from threading import Thread
import itertools
from collections import deque

# 3rd party lib
import numpy as np
import imagesize
from tqdm import tqdm


BG_OFFSET = -1
CROWD_IGNORED = False


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare OpenImages datset.')
    parser.add_argument('--data_dir', help='Directory path of OpenImages.')
    parser.add_argument('--src_name', help='Output annotation name.')
    parser.add_argument('--dst_name', help='Output annotation name.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    anno_dir = data_dir.joinpath('annotations')
    cocostyle_meta = json.load(open(anno_dir.joinpath(args.src_name), 'r'))

    # 1. process image information
    metas = []
    img_id2idx = {}
    for idx, img in tqdm(enumerate(cocostyle_meta['images'])):
        img_id2idx[img['id']] = idx
        img_meta = {}
        img_meta['filename'] = img['file_name']
        img_meta['height'] = img['height']
        img_meta['width'] = img['width']
        img_meta['ann'] = {'bboxes': [], 'labels': []}
        metas.append(img_meta)
    print('Images information processed.')

    # 2. process boxes information
    for inst in tqdm(cocostyle_meta['annotations']):
        img_id = inst['image_id']
        xywh_bbox = inst['bbox']
        x1,y1,w,h = xywh_bbox
        if w > 1 and h > 1:
            img_idx = img_id2idx[img_id]
            xyxy_bbox = [x1, y1, x1 + w, y1 + h]
            metas[img_idx]['ann']['bboxes'].append(xyxy_bbox)
            metas[img_idx]['ann']['labels'].append(inst['category_id'] + BG_OFFSET)
    print('Annotations information processed.')

    # 3. filter out images without annotations
    clean_metas = deque()
    for _ in tqdm(range(len(metas))):
        meta = metas.pop()
        bboxes = meta['ann']['bboxes']
        labels = meta['ann']['labels']
        if len(labels) > 0:
            clean_metas.appendleft(meta)
    print('Images with empty ground-truth removed.')

    # 4. store converted metas
    dst_path = anno_dir.joinpath(args.dst_name)
    with open(dst_path, 'w') as fout:
        json.dump(list(clean_metas), fout)
    print(f'Converted meta stored to `{dst_path}`.')


if __name__ == '__main__':
    main()
