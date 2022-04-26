# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import copy
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):

    fields = ["labels", "area", "iscrowd", "boxes", "track_ids", "masks"]

    def __init__(self,  img_folder, ann_file, transforms, norm_transforms,
                 return_masks=False, overflow_boxes=False, remove_no_obj_imgs=True,
                 prev_frame=False, prev_frame_rnd_augs=0.0, prev_prev_frame=False,
                 min_num_objects=0):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, overflow_boxes)

        annos_image_ids = [
            ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())]
        if remove_no_obj_imgs:
            self.ids = sorted(list(set(annos_image_ids)))

        if min_num_objects:
            counter = Counter(annos_image_ids)

            self.ids = [i for i in self.ids if counter[i] >= min_num_objects]

        self._prev_frame = prev_frame
        self._prev_frame_rnd_augs = prev_frame_rnd_augs
        self._prev_prev_frame = prev_prev_frame

    def _getitem_from_id(self, image_id, random_state=None, random_jitter=True):
        # if random state is given we do the data augmentation with the state
        # and then apply the random jitter. this ensures that (simulated) adjacent
        # frames have independent jitter.
        if random_state is not None:
            curr_random_state = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
            random.setstate(random_state['random'])
            torch.random.set_rng_state(random_state['torch'])

        img, target = super(CocoDetection, self).__getitem__(image_id)
        image_id = self.ids[image_id]
        target = {'image_id': image_id,
                  'annotations': target}
        img, target = self.prepare(img, target)

        if 'track_ids' not in target:
            target['track_ids'] = torch.arange(len(target['labels']))

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # ignore
        ignore = target.pop("ignore").bool()
        for field in self.fields:
            if field in target:
                target[f"{field}_ignore"] = target[field][ignore]
                target[field] = target[field][~ignore]

        if random_state is not None:
            random.setstate(curr_random_state['random'])
            torch.random.set_rng_state(curr_random_state['torch'])

        if random_jitter:
            img, target = self._add_random_jitter(img, target)
        img, target = self._norm_transforms(img, target)

        return img, target

    # TODO: add to the transforms and merge norm_transforms into transforms
    def _add_random_jitter(self, img, target):
        if self._prev_frame_rnd_augs:
            orig_w, orig_h = img.size

            crop_width = random.randint(
                int((1.0 - self._prev_frame_rnd_augs) * orig_w),
                orig_w)
            crop_height = int(orig_h * crop_width / orig_w)

            transform = T.RandomCrop((crop_height, crop_width))
            img, target = transform(img, target)

            img, target = T.resize(img, target, (orig_w, orig_h))

        return img, target

    # def _add_random_jitter(self, img, target):
    #     if self._prev_frame_rnd_augs: # and random.uniform(0, 1) < 0.5:
    #         orig_w, orig_h = img.size

    #         width, height = img.size
    #         size = random.randint(
    #             int((1.0 - self._prev_frame_rnd_augs) * min(width, height)),
    #             int((1.0 + self._prev_frame_rnd_augs) * min(width, height)))
    #         img, target = T.RandomResize([size])(img, target)

    #         width, height = img.size
    #         min_size = (
    #             int((1.0 - self._prev_frame_rnd_augs) * width),
    #             int((1.0 - self._prev_frame_rnd_augs) * height))
    #         transform = T.RandomSizeCrop(min_size=min_size)
    #         img, target = transform(img, target)

    #         width, height = img.size
    #         if orig_w < width:
    #             img, target = T.RandomCrop((height, orig_w))(img, target)
    #         else:
    #             total_pad = orig_w - width
    #             pad_left = random.randint(0, total_pad)
    #             pad_right = total_pad - pad_left

    #             padding = (pad_left, 0, pad_right, 0)
    #             img, target = T.pad(img, target, padding)

    #         width, height = img.size
    #         if orig_h < height:
    #             img, target = T.RandomCrop((orig_h, width))(img, target)
    #         else:
    #             total_pad = orig_h - height
    #             pad_top = random.randint(0, total_pad)
    #             pad_bottom = total_pad - pad_top

    #             padding = (0, pad_top, 0, pad_bottom)
    #             img, target = T.pad(img, target, padding)

    #     return img, target

    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}
        img, target = self._getitem_from_id(idx, random_state, random_jitter=False)

        if self._prev_frame:
            # PREV
            prev_img, prev_target = self._getitem_from_id(idx, random_state)
            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target

            if self._prev_prev_frame:
                # PREV PREV
                prev_prev_img, prev_prev_target = self._getitem_from_id(idx, random_state)
                target[f'prev_prev_image'] = prev_prev_img
                target[f'prev_prev_target'] = prev_prev_target

        return img, target

    def write_result_files(self, *args):
        pass


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if isinstance(polygons, dict):
            rles = {'size': polygons['size'],
                    'counts': polygons['counts'].encode(encoding='UTF-8')}
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, overflow_boxes=False):
        self.return_masks = return_masks
        self.overflow_boxes = overflow_boxes

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # x,y,w,h --> x,y,x,y
        boxes[:, 2:] += boxes[:, :2]
        if not self.overflow_boxes:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes - 1

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if anno and "track_id" in anno[0]:
            track_ids = torch.tensor([obj["track_id"] for obj in anno])
            target["track_ids"] = track_ids[keep]
        elif not len(boxes):
            target["track_ids"] = torch.empty(0)

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        ignore = torch.tensor([obj["ignore"] if "ignore" in obj else 0 for obj in anno])

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["ignore"] = ignore[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, img_transform=None, overflow_boxes=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # default
    max_size = 1333
    val_width = 800
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    random_resizes = [400, 500, 600]
    random_size_crop = (384, 600)

    if img_transform is not None:
        scale = img_transform.max_size / max_size
        max_size = img_transform.max_size
        val_width = img_transform.val_width

        # scale all with respect to custom max_size
        scales = [int(scale * s) for s in scales]
        random_resizes = [int(scale * s) for s in random_resizes]
        random_size_crop = [int(scale * s) for s in random_size_crop]

    if image_set == 'train':
        transforms = [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(random_resizes),
                    T.RandomSizeCrop(*random_size_crop, overflow_boxes=overflow_boxes),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
        ]
    elif image_set == 'val':
        transforms = [
            T.RandomResize([val_width], max_size=max_size),
        ]
    else:
        ValueError(f'unknown {image_set}')

    # transforms.append(normalize)
    return T.Compose(transforms), normalize


def build(image_set, args, mode='instances'):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    # image_set is 'train' or 'val'
    split = getattr(args, f"{image_set}_split")

    splits = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    if image_set == 'train':
        prev_frame_rnd_augs = args.coco_and_crowdhuman_prev_frame_rnd_augs
    elif image_set == 'val':
        prev_frame_rnd_augs = 0.0

    transforms, norm_transforms = make_coco_transforms(image_set, args.img_transform, args.overflow_boxes)
    img_folder, ann_file = splits[split]
    dataset = CocoDetection(
        img_folder, ann_file, transforms, norm_transforms,
        return_masks=args.masks,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame,
        min_num_objects=args.coco_min_num_objects)

    return dataset
