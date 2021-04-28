# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
CrowdHuman dataset with tracking training augmentations.
"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms


def build_crowdhuman(image_set, args):
    root = Path(args.crowdhuman_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / split
    ann_file = root / f'annotations/{split}.json'

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform)
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=transforms,
        norm_transforms=norm_transforms,
        return_masks=args.masks,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=args.coco_and_crowdhuman_prev_frame_rnd_augs)

    return dataset
