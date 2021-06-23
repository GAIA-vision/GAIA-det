# standard lib
import os
from pathlib import Path
import argparse
import json
from threading import Thread
import itertools

# 3rd party lib
from tqdm import tqdm
import imagesize


OID500_DET_META = 'challenge-2019-train-detection-bbox.csv'
OID500_CLS_DESCRIPTION = 'challenge-2019-classes-description-500.csv'
TAGS = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside"
TMAP = {t:i for i,t in enumerate(TAGS.strip().split(','))}
SPLIT = 'train'


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare OpenImages datset.')
    parser.add_argument('--oid_dir', help='Path of OpenImages directory.')
    parser.add_argument('--dst_name', help='Destination annotation name.')
    parser.add_argument('--nthread', type=int, default=64, help='Number of workers.')

    args = parser.parse_args()
    return args


def get_img_size_job(tid, img_dir, src, dst):
    img_metas = {}

    def get_img_size(img_id, img_dir):
        filename = img_id + '.jpg'
        img_path = img_dir.joinpath(filename)
        width, height = imagesize.get(img_path)
        img_metas[img_id] = {'filename': filename, 'width': width, 'height': height, 'lines': []}

    if tid == 0:
        for img_id in tqdm(src):
            get_img_size(img_id, img_dir)
    else:
        for img_id in src:
            get_img_size(img_id, img_dir)

    dst[tid].append(img_metas)


def get_img_size_manager(nthread, img_dir, src, dst):
    thread_list = []
    for tid in range(nthread):
        t_src = src[tid::nthread]
        thread = Thread(target=get_img_size_job, args=(tid, img_dir, t_src, dst))
        thread_list.append(thread)
        thread.start()

    for t in thread_list:
        t.join()


def process_lines_job(tid, img_ids, img_metas, cls_info, result_pool):

    def process_lines(img_id, img_metas, result_pool):
        dst_meta = {}
        dst_meta['ann'] = {}
        img_meta = img_metas[img_id]
        img_lines = img_meta['lines']

        dst_meta['filename'] = img_meta['filename']
        dst_meta['width'] = img_meta['width']
        dst_meta['height'] = img_meta['height']
        w = float(dst_meta['width'])
        h = float(dst_meta['height'])
        for line in img_lines:
            sp = line.strip().split(',')
            is_group = int(sp[TMAP['IsGroupOf']])
            x1, y1, x2, y2 = float(sp[TMAP['XMin']])*w, float(sp[TMAP['YMin']])*h, float(sp[TMAP['XMax']])*w, float(sp[TMAP['YMax']])*h
            label = cls_info[sp[TMAP['LabelName']]][0]
            if is_group or (y2 - y1) < 2 or (x2 - x1) < 2:
                bboxes_ignore = dst_meta['ann'].setdefault('bboxes_ignore', [])
                labels_ignore = dst_meta['ann'].setdefault('labels_ignore', [])
                bboxes_ignore.append([x1, y1, x2, y2])
                labels_ignore.append(label)
            else:
                bboxes = dst_meta['ann'].setdefault('bboxes', [])
                labels = dst_meta['ann'].setdefault('labels', [])
                bboxes.append([x1, y1, x2, y2])
                labels.append(label)
        if dst_meta['ann'].get('bboxes', None):
            result_pool[tid].append(dst_meta)

    if tid == 0:
        for img_id in tqdm(img_ids):
            process_lines(img_id, img_metas, result_pool)
    else:
        for img_id in img_ids:
            process_lines(img_id, img_metas, result_pool)


def process_lines_manager(nthread, img_ids, img_metas, cls_info, result_pool):
    thread_list = []
    for tid in range(nthread):
        img_id_per_thread = img_ids[tid::nthread]
        thread = Thread(target=process_lines_job, args=(
            tid, img_id_per_thread, img_metas, cls_info, result_pool))
        thread_list.append(thread)
        thread.start()

    for t in thread_list:
        t.join()


def main():
    args = parse_args()
    data_dir = Path(args.oid_dir)
    nthread = args.nthread
    assert data_dir.exists() and data_dir.is_dir()
    img_dir = data_dir.joinpath('images').joinpath(SPLIT)
    anno_dir = data_dir.joinpath('annotations')

    # 1. get image size
    det_meta = anno_dir.joinpath(OID500_DET_META)
    img_id_set = set()
    with open(det_meta, 'r') as fin:
        header = None
        for l in fin:
            if not header:
                header = l
                continue
            img_id = l.strip().split(',')[0]
            img_id_set.add(img_id)

    result_pool = [[] for _ in range(nthread)]
    img_id_set = tuple(img_id_set)
    get_img_size_manager(nthread, img_dir, img_id_set, result_pool)
    img_metas = {}
    for d in tqdm(result_pool):
        img_metas.update(d[0])
    print('Image information collected.')

    # 2. collect class info
    cls_info = {}
    with open(anno_dir.joinpath(OID500_CLS_DESCRIPTION), 'r') as fin:
        for i, l in enumerate(fin):
            cls_id, cls_name = l.strip().split(',')
            cls_info[cls_id] = (i, cls_name)
    print('Class information collected.')

    with open(det_meta, 'r') as fin:
        header = None
        for l in tqdm(fin):
            if not header:
                header = l
                continue
            img_id = l.split(',', 1)[0]
            img_metas[img_id]['lines'].append(l)
    print('Box information collected.')

    # 3. prepare box info
    img_ids = tuple(img_metas.keys())
    result_pool = [[] for _ in range(nthread)]
    process_lines_manager(nthread, img_ids, img_metas, cls_info, result_pool)
    oid500_meta = list(itertools.chain(*result_pool))
    print('OpenImages(Challenge 2019) dataset processed.')

    # 4. store converted meta
    dst_path = anno_dir.joinpath(args.dst_name)
    with open(dst_path, 'w') as fout:
        json.dump(oid500_meta, fout)
    print(f'Converted meta stored to `{dst_path}`.')


if __name__ == '__main__':
    main()
