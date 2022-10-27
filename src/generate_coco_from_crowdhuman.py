# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Generates COCO data and annotation structure from CrowdHuman data.
"""
import json
import os
import cv2

from generate_coco_from_mot import check_coco_from_mot

DATA_ROOT = 'data/CrowdHuman'
VIS_THRESHOLD = 0.0


def generate_coco_from_crowdhuman(split_name='train_val', split='train_val'):
    """
    Generate COCO data from CrowdHuman.
    """
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "person",
                                  "name": "person",
                                  "id": 1}]
    annotations['annotations'] = []
    annotation_file = os.path.join(DATA_ROOT, f'annotations/{split_name}.json')

    # IMAGES
    imgs_list_dir = os.listdir(os.path.join(DATA_ROOT, split))
    for i, img in enumerate(sorted(imgs_list_dir)):
        im = cv2.imread(os.path.join(DATA_ROOT, split, img))
        h, w, _ = im.shape

        annotations['images'].append({
            "file_name": img,
            "height": h,
            "width": w,
            "id": i, })

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        os.path.splitext(img_dict['file_name'])[0]: img_dict['id']
        for img_dict in annotations['images']}

    for split in ['train', 'val']:
        if split not in split_name:
            continue
        odgt_annos_file = os.path.join(DATA_ROOT, f'annotations/annotation_{split}.odgt')
        with open(odgt_annos_file, 'r+') as anno_file:
            datalist = anno_file.readlines()

        ignores = 0
        for data in datalist:
            json_data = json.loads(data)
            gtboxes = json_data['gtboxes']
            for gtbox in gtboxes:
                if gtbox['tag'] == 'person':
                    bbox = gtbox['fbox']
                    area = bbox[2] * bbox[3]

                    ignore = False
                    visibility = 1.0
                    # if 'occ' in gtbox['extra']:
                    #     visibility = 1.0 - gtbox['extra']['occ']
                    # if visibility <= VIS_THRESHOLD:
                    #     ignore = True

                    if 'ignore' in gtbox['extra']:
                        ignore = ignore or bool(gtbox['extra']['ignore'])

                    ignores += int(ignore)

                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": img_file_name_to_id[json_data['ID']],
                        "segmentation": [],
                        "ignore": int(ignore),
                        "visibility": visibility,
                        "area": area,
                        "iscrowd": 0,
                        "category_id": annotations['categories'][0]['id'],}

                    annotation_id += 1
                    annotations['annotations'].append(annotation)

    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]
        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max([n for n  in num_objs_per_image.values()])}')
    print(f'ignore augs: {ignores}/{len(annotations["annotations"])}')
    print(len(annotations['images']))

    # for img_id, num_objs in num_objs_per_image.items():
    #     if num_objs > 50 or num_objs < 2:
    #         annotations['images'] = [
    #             img for img in annotations['images']
    #             if img_id != img['id']]

    #         annotations['annotations'] = [
    #             anno for anno in annotations['annotations']
    #             if img_id != anno['image_id']]

    # print(len(annotations['images']))

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


if __name__ == '__main__':
    generate_coco_from_crowdhuman(split_name='train_val', split='train_val')
    # generate_coco_from_crowdhuman(split_name='train', split='train')

    # coco_dir = os.path.join('data/CrowdHuman', 'train_val')
    # annotation_file = os.path.join('data/CrowdHuman/annotations', 'train_val.json')
    # check_coco_from_mot(coco_dir, annotation_file, img_id=9012)
