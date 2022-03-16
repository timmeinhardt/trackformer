# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Generates COCO data and annotation structure from MOTChallenge data.
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import numpy as np
import pycocotools.mask as rletools
import skimage.io as io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou

from trackformer.datasets.tracking.mots20_sequence import load_mots_gt

MOTS_ROOT = 'data/MOTS20'
VIS_THRESHOLD = 0.25

MOT_15_SEQS_INFO = {
    'ETH-Bahnhof': {'img_width': 640, 'img_height': 480, 'seq_length': 1000},
    'ETH-Sunnyday': {'img_width': 640, 'img_height': 480, 'seq_length': 354},
    'KITTI-13': {'img_width': 1242, 'img_height': 375, 'seq_length': 340},
    'KITTI-17': {'img_width': 1224, 'img_height': 370, 'seq_length': 145},
    'PETS09-S2L1': {'img_width': 768, 'img_height': 576, 'seq_length': 795},
    'TUD-Campus': {'img_width': 640, 'img_height': 480, 'seq_length': 71},
    'TUD-Stadtmitte': {'img_width': 640, 'img_height': 480, 'seq_length': 179},}


def generate_coco_from_mot(split_name='train', seqs_names=None,
                           root_split='train', mots=False, mots_vis=False,
                           frame_range=None, data_root='data/MOT17'):
    """
    Generates COCO data from MOT.
    """
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    if mots:
        data_root = MOTS_ROOT
    root_split_path = os.path.join(data_root, root_split)
    root_split_mots_path = os.path.join(MOTS_ROOT, root_split)
    coco_dir = os.path.join(data_root, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "person",
                                  "name": "person",
                                  "id": 1}]
    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0

    seqs = sorted(os.listdir(root_split_path))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, seqs)

    for seq in seqs:
        # CONFIG FILE
        config = configparser.ConfigParser()
        config_file = os.path.join(root_split_path, seq, 'seqinfo.ini')

        if os.path.isfile(config_file):
            config.read(config_file)
            img_width = int(config['Sequence']['imWidth'])
            img_height = int(config['Sequence']['imHeight'])
            seq_length = int(config['Sequence']['seqLength'])
        else:
            img_width = MOT_15_SEQS_INFO[seq]['img_width']
            img_height = MOT_15_SEQS_INFO[seq]['img_height']
            seq_length = MOT_15_SEQS_INFO[seq]['seq_length']

        seg_list_dir = os.listdir(os.path.join(root_split_path, seq, 'img1'))
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)

        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id

            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            img_id += 1

            os.symlink(os.path.join(os.getcwd(), root_split_path, seq, 'img1', img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        img_dict['file_name']: img_dict['id']
        for img_dict in annotations['images']}
    for seq in seqs:
        # GT FILE
        gt_file_path = os.path.join(root_split_path, seq, 'gt', 'gt.txt')
        if mots:
            gt_file_path = os.path.join(
                root_split_mots_path,
                seq.replace('MOT17', 'MOTS20'),
                'gt',
                'gt.txt')
        if not os.path.isfile(gt_file_path):
            continue

        seq_annotations = []
        if mots:
            mask_objects_per_frame = load_mots_gt(gt_file_path)
            for frame_id, mask_objects in mask_objects_per_frame.items():
                for mask_object in mask_objects:
                    # class_id = 1 is car
                    # class_id = 2 is person
                    # class_id = 10 IGNORE
                    if mask_object.class_id == 1:
                        continue

                    bbox = rletools.toBbox(mask_object.mask)
                    bbox = [int(c) for c in bbox]
                    area = bbox[2] * bbox[3]
                    image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.jpg", None)
                    if image_id is None:
                        continue

                    segmentation = {
                        'size': mask_object.mask['size'],
                        'counts': mask_object.mask['counts'].decode(encoding='UTF-8')}

                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": image_id,
                        "segmentation": segmentation,
                        "ignore": mask_object.class_id == 10,
                        "visibility": 1.0,
                        "area": area,
                        "iscrowd": 0,
                        "seq": seq,
                        "category_id": annotations['categories'][0]['id'],
                        "track_id": mask_object.track_id}

                    seq_annotations.append(annotation)
                    annotation_id += 1

            annotations['annotations'].extend(seq_annotations)
        else:

            seq_annotations_per_frame = {}
            with open(gt_file_path, "r") as gt_file:
                reader = csv.reader(gt_file, delimiter=' ' if mots else ',')

                for row in reader:
                    if int(row[6]) == 1 and (seq in MOT_15_SEQS_INFO or int(row[7]) == 1):
                        bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                        bbox = [int(c) for c in bbox]

                        area = bbox[2] * bbox[3]
                        visibility = float(row[8])
                        frame_id = int(row[0])
                        image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.jpg", None)
                        if image_id is None:
                            continue
                        track_id = int(row[1])

                        annotation = {
                            "id": annotation_id,
                            "bbox": bbox,
                            "image_id": image_id,
                            "segmentation": [],
                            "ignore": 0 if visibility > VIS_THRESHOLD else 1,
                            "visibility": visibility,
                            "area": area,
                            "iscrowd": 0,
                            "seq": seq,
                            "category_id": annotations['categories'][0]['id'],
                            "track_id": track_id}

                        seq_annotations.append(annotation)
                        if frame_id not in seq_annotations_per_frame:
                            seq_annotations_per_frame[frame_id] = []
                        seq_annotations_per_frame[frame_id].append(annotation)

                        annotation_id += 1

            annotations['annotations'].extend(seq_annotations)

            #change ignore based on MOTS mask
            if mots_vis:
                gt_file_mots = os.path.join(
                    root_split_mots_path,
                    seq.replace('MOT17', 'MOTS20'),
                    'gt',
                    'gt.txt')
                if os.path.isfile(gt_file_mots):
                    mask_objects_per_frame = load_mots_gt(gt_file_mots)

                    for frame_id, frame_annotations in seq_annotations_per_frame.items():
                        mask_objects = mask_objects_per_frame[frame_id]
                        mask_object_bboxes = [rletools.toBbox(obj.mask) for obj in mask_objects]
                        mask_object_bboxes = torch.tensor(mask_object_bboxes).float()

                        frame_boxes = [a['bbox'] for a in frame_annotations]
                        frame_boxes = torch.tensor(frame_boxes).float()

                        # x,y,w,h --> x,y,x,y
                        frame_boxes[:, 2:] += frame_boxes[:, :2]
                        mask_object_bboxes[:, 2:] += mask_object_bboxes[:, :2]

                        mask_iou = box_iou(mask_object_bboxes, frame_boxes)

                        mask_indices, frame_indices = linear_sum_assignment(-mask_iou)
                        for m_i, f_i in zip(mask_indices, frame_indices):
                            if mask_iou[m_i, f_i] < 0.5:
                                continue

                            if not frame_annotations[f_i]['visibility']:
                                frame_annotations[f_i]['ignore'] = 0

    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


def check_coco_from_mot(coco_dir='data/MOT17/mot17_train_coco', annotation_file='data/MOT17/annotations/mot17_train_coco.json', img_id=None):
    """
    Visualize generated COCO data. Only used for debugging.
    """
    # coco_dir = os.path.join(data_root, split)
    # annotation_file = os.path.join(coco_dir, 'annotations.json')

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    if img_id == None:
        img_ids = coco.getImgIds(catIds=cat_ids)
        index = np.random.randint(0, len(img_ids))
        img_id = img_ids[index]
    img = coco.loadImgs(img_id)[0]

    i = io.imread(os.path.join(coco_dir, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig('annotations.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO from MOT.')
    parser.add_argument('--mots20', action='store_true')
    parser.add_argument('--mot20', action='store_true')
    args = parser.parse_args()

    mot15_seqs_names = list(MOT_15_SEQS_INFO.keys())

    if args.mots20:
        #
        # MOTS20
        #

        # TRAIN SET
        generate_coco_from_mot(
            'mots20_train_coco',
            seqs_names=['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11'],
            mots=True)

        # TRAIN SPLITS
        for i in range(4):
            train_seqs = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
            val_seqs = train_seqs.pop(i)

            generate_coco_from_mot(
                f'mots20_train_{i + 1}_coco',
                seqs_names=train_seqs, mots=True)
            generate_coco_from_mot(
                f'mots20_val_{i + 1}_coco',
                seqs_names=val_seqs, mots=True)

    elif args.mot20:
        data_root = 'data/MOT20'
        train_seqs = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',]
        # TRAIN SET
        generate_coco_from_mot(
            'mot20_train_coco',
            seqs_names=train_seqs,
            data_root=data_root)

        for i in range(0, len(train_seqs)):
            train_seqs_copy = train_seqs.copy()
            val_seqs = train_seqs_copy.pop(i)

            generate_coco_from_mot(
                f'mot20_train_{i + 1}_coco',
                seqs_names=train_seqs_copy,
                data_root=data_root)
            generate_coco_from_mot(
                f'mot20_val_{i + 1}_coco',
                seqs_names=val_seqs,
                data_root=data_root)

        # CROSS VAL FRAME SPLIT
        generate_coco_from_mot(
            'mot20_train_cross_val_frame_0_0_to_0_5_coco',
            seqs_names=train_seqs,
            frame_range={'start': 0, 'end': 0.5},
            data_root=data_root)
        generate_coco_from_mot(
            'mot20_train_cross_val_frame_0_5_to_1_0_coco',
            seqs_names=train_seqs,
            frame_range={'start': 0.5, 'end': 1.0},
            data_root=data_root)

    else:
        #
        # MOT17
        #

        # CROSS VAL SPLIT 1
        generate_coco_from_mot(
            'mot17_train_cross_val_1_coco',
            seqs_names=['MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-11-FRCNN'])
        generate_coco_from_mot(
            'mot17_val_cross_val_1_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-10-FRCNN', 'MOT17-13-FRCNN'])

        # CROSS VAL SPLIT 2
        generate_coco_from_mot(
            'mot17_train_cross_val_2_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-13-FRCNN'])
        generate_coco_from_mot(
            'mot17_val_cross_val_2_coco',
            seqs_names=['MOT17-04-FRCNN', 'MOT17-11-FRCNN'])

        # CROSS VAL SPLIT 3
        generate_coco_from_mot(
            'mot17_train_cross_val_3_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'])
        generate_coco_from_mot(
            'mot17_val_cross_val_3_coco',
            seqs_names=['MOT17-05-FRCNN', 'MOT17-09-FRCNN'])

        # CROSS VAL FRAME SPLIT
        generate_coco_from_mot(
            'mot17_train_cross_val_frame_0_0_to_0_25_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'],
            frame_range={'start': 0, 'end': 0.25})
        generate_coco_from_mot(
            'mot17_train_cross_val_frame_0_0_to_0_5_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'],
            frame_range={'start': 0, 'end': 0.5})
        generate_coco_from_mot(
            'mot17_train_cross_val_frame_0_5_to_1_0_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'],
            frame_range={'start': 0.5, 'end': 1.0})

        generate_coco_from_mot(
            'mot17_train_cross_val_frame_0_75_to_1_0_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'],
            frame_range={'start': 0.75, 'end': 1.0})

        # TRAIN SET
        generate_coco_from_mot(
            'mot17_train_coco',
            seqs_names=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
                        'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'])

        for i in range(0, 7):
            train_seqs = [
                'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
                'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
            val_seqs = train_seqs.pop(i)

            generate_coco_from_mot(
                f'mot17_train_{i + 1}_coco',
                seqs_names=train_seqs)
            generate_coco_from_mot(
                f'mot17_val_{i + 1}_coco',
                seqs_names=val_seqs)
